"""
enrich_mechanisms.py — generates GSTT1-style "why" mechanism narratives for
every gene/variant with an ADR-linked edge in the graph, and writes them into
kg_export/edges.json as a new `mechanism_narrative` property.

Kept separate from kg.py because it depends on two external services kg.py
doesn't need: UniProt REST (gene function grounding) and a local Ollama
endpoint (narrative synthesis) — and because the two need different
environments on this cluster:

    UniProt needs internet     -> run on the LOGIN node
    Ollama needs a GPU + no internet required -> run on a COMPUTE node

So this is five separate resumable, cacheable steps rather than one command:

    python enrich_mechanisms.py fetch-gene-functions   # login node (internet)
    python enrich_mechanisms.py draft-adr-pathways     # compute node (Ollama)
    python enrich_mechanisms.py synthesize             # compute node (Ollama)
    python enrich_mechanisms.py apply                  # either (no network/LLM)
    python enrich_mechanisms.py qa-sample              # either (no network/LLM)

After `apply`, run `python kg.py load` to push the enriched edges.json into
Neo4j.

Every step is cache-backed under kg_export/ (gene_function_cache.json,
adr_pathway_cache.json, mechanism_narrative_cache.json) and safe to
interrupt/rerun — already-cached entries are skipped, and progress is
flushed to disk periodically, not just at the end.

Scope: only the ADR-linked edge types (LINKED_TO_ADR, ASSOCIATED_WITH_ADR,
CLINVAR_ASSOCIATED_ADR) get a narrative — that's the "this variant causes
this ADR, here's why" shape the GSTT1 example is. Drug-response/efficacy
edges are out of scope for this pass.
"""

import argparse
import json
import os
import random
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict

from openai import OpenAI

from kg import CTCAE_MAP  # reuse the single source of truth for canonical ADR names

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(SCRIPT_DIR, "kg_export")
NODES_PATH = os.path.join(EXPORT_DIR, "nodes.json")
EDGES_PATH = os.path.join(EXPORT_DIR, "edges.json")

GENE_FUNCTION_CACHE = os.path.join(EXPORT_DIR, "gene_function_cache.json")
ADR_PATHWAY_CACHE = os.path.join(EXPORT_DIR, "adr_pathway_cache.json")
NARRATIVE_CACHE = os.path.join(EXPORT_DIR, "mechanism_narrative_cache.json")
QA_SAMPLE_PATH = os.path.join(EXPORT_DIR, "mechanism_narrative_qa_sample.md")

ADR_LINKED_RELATIONS = {"LINKED_TO_ADR", "ASSOCIATED_WITH_ADR", "CLINVAR_ASSOCIATED_ADR"}
CANONICAL_ADRS = list(CTCAE_MAP.keys())

BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
MODEL = os.environ.get("LLM_MODEL", "qwen3:8b")

QA_HIGHLIGHT_GENES = {"GSTT1", "TPMT", "CYP2D6", "GSTM1"}

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"


# ── Shared JSON cache helpers ──────────────────────────────────

def _load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_graph():
    nodes = _load_json(NODES_PATH, default=None)
    edges = _load_json(EDGES_PATH, default=None)
    if nodes is None or edges is None:
        raise SystemExit(f"Missing {NODES_PATH} or {EDGES_PATH} — run this from OncologyKG/.")
    return nodes, edges


def _llm_client():
    return OpenAI(base_url=BASE_URL, api_key="ollama")


# ── Gene resolution + tuple collection ─────────────────────────
# Every Variant node already carries a gene_symbols property (see kg.py's
# parse_variants) — that's the primary source. Star alleles/named variants
# (e.g. "CYP2D6*4", "GSTT1 null") fall back to a first-token split. Variants
# with neither (mostly bare rsIDs) are left unmapped rather than guessed.

import re

_RSID_RE = re.compile(r"^rs\d+$", re.IGNORECASE)


def _variant_to_gene(nodes):
    mapping = {}
    for n in nodes:
        if n["label"] != "Variant":
            continue
        p = n["properties"]
        name = p["name"]
        gene_symbols = p.get("gene_symbols", "")
        # pandas' NaN gets stringified to the literal text "nan" by kg.py's
        # str(row.get(...)) pattern — that's a missing value, not a gene symbol.
        if gene_symbols and gene_symbols.strip().lower() != "nan":
            mapping[name] = gene_symbols.split(",")[0].strip()
        elif "*" in name:
            mapping[name] = name.split("*")[0].strip()
        elif not _RSID_RE.match(name):
            # e.g. "GSTT1 null" / "GSTT1 non-null" — first token is the gene.
            # A bare rsID has no such structure, so it's left unmapped rather
            # than treating the rsID itself as if it were a gene symbol.
            mapping[name] = name.split(" ")[0].strip()
    return mapping


def _edge_direction(props):
    return props.get("direction") or props.get("side_effect_type") or ""


def _cache_key(variant, adr):
    return f"{variant}|||{adr}"


def collect_adr_tuples(nodes, edges):
    """Returns {(variant, gene, adr): [edge indices]} for every ADR-linked
    edge that resolves to a gene.

    Keyed by VARIANT, not gene: two variants of the same gene can be
    biologically opposite (e.g. "GSTT1 null" vs "GSTT1 non-null" — a
    deletion vs a normal-function allele). Deduping on gene alone collapses
    those into one narrative that can't say which genotype it's actually
    describing.

    NOT keyed by direction, even though a given (variant, adr) pair can have
    multiple edges (e.g. LINKED_TO_ADR from clinicalVariants.tsv AND
    ASSOCIATED_WITH_ADR from variantAnnotations.tsv) with different direction/
    side_effect_type property shapes — those are almost always the exact same
    underlying PharmGKB finding cited from two source tables (confirmed: GSTT1
    null's LINKED_TO_ADR and ASSOCIATED_WITH_ADR edges to Ototoxicity carry
    identical description text). kg.py's enrich_with_cross_referenced_mechanism
    already treats these as one fact for description/mechanism purposes; doing
    the same here avoids synthesizing near-duplicate narratives for what's
    really a single finding. See _representative_direction_and_description,
    which picks one direction/description across all edges in the group.
    """
    variant_gene = _variant_to_gene(nodes)
    tuples_to_edges = defaultdict(list)
    for idx, e in enumerate(edges):
        if e["relation"] not in ADR_LINKED_RELATIONS:
            continue
        if e["head_label"] == "Variant":
            variant, adr = e["head"], e["tail"]
        elif e["tail_label"] == "Variant":
            variant, adr = e["tail"], e["head"]
        else:
            continue
        gene = variant_gene.get(variant)
        if not gene:
            continue
        tuples_to_edges[(variant, gene, adr)].append(idx)
    return tuples_to_edges


def _representative_direction_and_description(edges, idxs):
    direction = ""
    description = ""
    for idx in idxs:
        props = edges[idx]["properties"]
        if not direction:
            direction = _edge_direction(props)
        if not description:
            description = props.get("description", "")
    return direction, description


# ── Step 1: UniProt gene function fetch (login node, needs internet) ───

def fetch_gene_function(gene_symbol, timeout=15):
    query = f"gene:{gene_symbol} AND organism_id:9606 AND reviewed:true"
    params = urllib.parse.urlencode({
        "query": query,
        "fields": "accession,cc_function",
        "format": "json",
        "size": 1,
    })
    url = f"{UNIPROT_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "OncologyKG-enrichment/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    results = data.get("results", [])
    if not results:
        return None
    texts = []
    for c in results[0].get("comments", []):
        if c.get("commentType") == "FUNCTION":
            texts.extend(t["value"] for t in c.get("texts", []))
    return " ".join(texts) if texts else None


def cmd_fetch_gene_functions():
    nodes, edges = load_graph()
    tuples_to_edges = collect_adr_tuples(nodes, edges)
    genes_needed = sorted({gene for (variant, gene, adr) in tuples_to_edges})
    print(f"{len(genes_needed)} distinct genes need function text "
          f"(from {len(tuples_to_edges):,} unique variant/ADR/direction tuples)")

    cache = _load_json(GENE_FUNCTION_CACHE, default={})
    fetched = no_hit = failed = skipped = 0
    for i, gene in enumerate(genes_needed):
        if gene in cache:
            skipped += 1
            continue
        try:
            fn = fetch_gene_function(gene)
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            print(f"  [{i + 1}/{len(genes_needed)}] {gene}: ERROR {e}")
            failed += 1
            continue
        if fn:
            cache[gene] = fn
            fetched += 1
            print(f"  [{i + 1}/{len(genes_needed)}] {gene}: OK ({len(fn)} chars)")
        else:
            cache[gene] = None  # explicit "checked, no reviewed hit" marker — don't refetch forever
            no_hit += 1
            print(f"  [{i + 1}/{len(genes_needed)}] {gene}: no reviewed UniProt entry")
        if (i + 1) % 20 == 0:
            _save_json(GENE_FUNCTION_CACHE, cache)
        time.sleep(0.34)  # ~3 req/s — polite to a shared public API

    _save_json(GENE_FUNCTION_CACHE, cache)
    print(f"\nDone. fetched={fetched} no_hit={no_hit} failed={failed} "
          f"already_cached={skipped} (cache now has {len(cache)} entries)")
    if failed:
        print("Rerun this command to retry failed lookups — cached entries are skipped.")


# ── Step 2: ADR pathway backbone (compute node, needs Ollama) ──────

ADR_DRAFT_PROMPT_TEMPLATE = """You are a clinical pharmacologist. In 2-3 plain-language sentences, \
describe the general biochemical/cellular injury pathway by which chemotherapy \
drugs commonly cause {adr} in patients — focus on the mechanism of tissue/organ \
damage itself (e.g. reactive oxygen species, direct DNA damage, receptor-mediated \
effects), not any single specific drug or gene. Write it so it could plug into the \
middle of a sentence like "...this leads to X, which damages Y, causing {adr}." \
Do not mention any specific gene or drug name. Output ONLY the sentences, no \
preamble, no headers."""


def cmd_draft_adr_pathways():
    cache = _load_json(ADR_PATHWAY_CACHE, default={})
    client = _llm_client()
    for adr in CANONICAL_ADRS:
        if adr in cache:
            continue
        prompt = ADR_DRAFT_PROMPT_TEMPLATE.format(adr=adr)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text = resp.choices[0].message.content.strip()
        cache[adr] = text
        print(f"{adr}:\n  {text}\n")
        _save_json(ADR_PATHWAY_CACHE, cache)
    print(f"Done. {len(cache)}/{len(CANONICAL_ADRS)} ADR pathway paragraphs drafted.")
    print(f"Skim {ADR_PATHWAY_CACHE} once before running `synthesize` — cheap to hand-edit if anything looks off.")


# ── Step 3: per-(gene, ADR, direction) narrative synthesis (compute node) ──

NARRATIVE_STYLE_EXAMPLE = """EXAMPLE STYLE when the mechanism IS established — copy this structure and \
tone exactly, but NEVER copy these specific facts (they are for GSTT1/cisplatin/ototoxicity \
only; your gene and ADR will usually be different):

"Looking at the GSTT1 null gene variant genotype, where the gene is deleted. This \
leads to reduced production of GSTT1 enzyme, which would normally help neutralize \
harmful byproducts of cisplatin. Since there is less of that enzyme around, the \
toxic metabolites don't get removed as efficiently and stay around longer, which \
leads to the higher risk of hearing loss."

EXAMPLE STYLE when the mechanism is NOT established — shorter, states the real \
function and the real association, then says plainly that no mechanism connects \
them (this is the correct output whenever the gene's function has no textbook \
link to the specific drug involved — do not stretch the function to fit anyway):

"Looking at the TPMT rs1142345 variant (allele C), the knowledge graph records an \
association with increased ototoxicity risk when treated with cisplatin. TPMT's \
established function is methylating thiopurine drugs like 6-mercaptopurine, which \
has no known connection to cisplatin or platinum-based chemotherapy — the \
biological mechanism behind this association is not established in the evidence \
available." """


# These narratives are meant to be short (1-4 sentences), so 500 tokens is
# generous headroom — set explicitly rather than trusting Ollama's default,
# since a real narrative was observed truncated mid-word ("...displaying
# glutath,") in testing. If it still gets cut off, retry once before giving up
# rather than caching/returning a broken sentence.
def _generate_narrative(client, prompt, model=None):
    model = model or MODEL
    for attempt in range(2):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
        )
        choice = resp.choices[0]
        text = choice.message.content.strip()
        if choice.finish_reason != "length":
            return text
        print(f"  (truncated at finish_reason=length, retrying, attempt {attempt + 1})")
    return text  # both attempts truncated — return what we have rather than lose it entirely


def _direction_clause(direction, variant_names, description):
    parts = [f"Variant/genotype name(s) in the knowledge graph: {', '.join(variant_names)}."]
    if direction:
        parts.append(f"Direction of effect recorded in the knowledge graph: {direction} risk/likelihood/severity.")
    if description:
        parts.append(f'Specific finding recorded in the knowledge graph: "{description}"')
    return " ".join(parts)


def build_narrative_prompt(gene, function_text, adr, adr_pathway, direction, variant_names, description):
    clause = _direction_clause(direction, variant_names, description)
    return f"""You are a clinical pharmacogenomicist explaining, in plain language for a \
parent or clinician (not a specialist audience), WHY a genetic variant changes the \
risk of a drug side effect.

{NARRATIVE_STYLE_EXAMPLE}

Now write ONE for this case:

Gene: {gene}
Real biological function of this gene's protein (source: UniProt): {function_text}

Adverse drug reaction: {adr}
General injury pathway for this ADR: {adr_pathway}

{clause}

Before writing: check whether the gene's real function (given above) actually connects to \
the drug named in the knowledge-graph finding through an established, textbook \
pharmacological pathway (e.g. the gene's protein is a known enzyme/transporter for \
that drug or a close relative of it). Do NOT invent a novel biochemical bridge \
connecting an unrelated function to this drug just to complete a causal chain — \
e.g. a thiopurine-metabolizing enzyme showing a statistical association with an \
unrelated drug's toxicity is NOT license to invent a pathway where that drug's \
"metabolites" build up because of it; that link is not established by anything \
above, and stating it as fact would be fabricating pharmacology.

- If the connection IS established: write a single flowing paragraph of 2-4 \
  sentences following the example's causal-chain structure: name the \
  variant/genotype -> what it does to the gene's normal function -> the \
  resulting biochemical consequence -> how that leads to {adr}.
- If it is NOT established (the gene's real function has no clear textbook link \
  to this specific drug, or the knowledge-graph finding gives only a bare \
  statistical association with no explanatory mechanism): write 1-2 sentences \
  instead — name the variant/genotype, state plainly what the knowledge graph \
  found (the association with {adr}), state the gene's real function, and say \
  directly that the biological mechanism connecting them is not established in \
  the evidence available. STOP THERE. Do not add a further sentence that \
  speculates how they might still be connected anyway ("may alter...", "could \
  lead to...", "if the variant affects...", "potentially...") — hedged \
  speculation is still fabrication, just softened, and defeats the entire point \
  of this branch. The paragraph ends at "the mechanism is not established"; \
  nothing about a possible pathway comes after that.

Either way, base every biological claim ONLY on the gene function and ADR pathway text \
given above and the knowledge-graph finding quoted above — do not invent \
additional biology, and do not name a different drug than the one in the KG \
finding (or omit the drug entirely if none is given). Stay tightly focused on \
explaining this specific {adr} finding — the gene function text above may mention \
other substrates, other drugs, or broader biology; include only the part of it \
that's actually needed for this causal chain, and leave the rest out entirely. \
A shorter, tightly-relevant paragraph is better than a longer one padded with \
gene trivia that doesn't bear on this specific case. Output ONLY the paragraph \
itself — no headers, no preamble, no quotation marks around it."""


def cmd_synthesize():
    nodes, edges = load_graph()
    tuples_to_edges = collect_adr_tuples(nodes, edges)
    gene_functions = _load_json(GENE_FUNCTION_CACHE, default={})
    adr_pathways = _load_json(ADR_PATHWAY_CACHE, default={})
    if len(adr_pathways) < len(CANONICAL_ADRS):
        raise SystemExit("Run `draft-adr-pathways` first — not all 10 ADR pathways are cached yet.")
    narrative_cache = _load_json(NARRATIVE_CACHE, default={})

    client = _llm_client()
    todo = [t for t in tuples_to_edges if _cache_key(t[0], t[2]) not in narrative_cache]
    print(f"{len(tuples_to_edges):,} total tuples, {len(todo):,} remaining to synthesize")

    no_grounding = 0
    for i, (variant, gene, adr) in enumerate(todo):
        function_text = gene_functions.get(gene)
        if not function_text:
            no_grounding += 1
            continue  # no UniProt hit for this gene — OncologyKGMM.py falls back
                      # to the model's own general knowledge for these at answer time

        direction, description = _representative_direction_and_description(
            edges, tuples_to_edges[(variant, gene, adr)]
        )

        prompt = build_narrative_prompt(
            gene, function_text, adr, adr_pathways[adr], direction,
            [variant], description,
        )
        try:
            text = _generate_narrative(client, prompt)
        except Exception as e:
            print(f"  [{i + 1}/{len(todo)}] {variant}/{adr}: ERROR {e}")
            continue

        narrative_cache[_cache_key(variant, adr)] = text
        print(f"  [{i + 1}/{len(todo)}] {variant}/{adr}: {text[:80]}...")
        if (i + 1) % 10 == 0:
            _save_json(NARRATIVE_CACHE, narrative_cache)

    _save_json(NARRATIVE_CACHE, narrative_cache)
    print(f"\nDone. {len(narrative_cache):,} narratives cached total. "
          f"{no_grounding} tuples skipped (no UniProt gene function available).")


# ── Step 4: write cached narratives into edges.json ─────────────

def cmd_apply():
    nodes, edges = load_graph()
    tuples_to_edges = collect_adr_tuples(nodes, edges)
    narrative_cache = _load_json(NARRATIVE_CACHE, default={})

    applied = 0
    for (variant, gene, adr), idxs in tuples_to_edges.items():
        narrative = narrative_cache.get(_cache_key(variant, adr))
        if not narrative:
            continue
        for idx in idxs:
            edges[idx]["properties"]["mechanism_narrative"] = narrative
            applied += 1

    backup_path = EDGES_PATH + ".bak"
    shutil.copy(EDGES_PATH, backup_path)
    _save_json(EDGES_PATH, edges)
    print(f"Applied mechanism_narrative to {applied:,} edges (backup saved to {backup_path})")
    print("Run `python kg.py load` next to push the enriched graph into Neo4j.")


# ── Step 5: QA sample for manual review ─────────────────────────

def cmd_qa_sample():
    nodes, edges = load_graph()
    tuples_to_edges = collect_adr_tuples(nodes, edges)
    narrative_cache = _load_json(NARRATIVE_CACHE, default={})

    highlight, rest = [], []
    for t in tuples_to_edges:
        variant, gene, adr = t
        if _cache_key(variant, adr) not in narrative_cache:
            continue
        (highlight if gene in QA_HIGHLIGHT_GENES else rest).append(t)

    random.seed(42)
    sample = highlight + random.sample(rest, min(30, len(rest)))

    lines = ["# Mechanism narrative QA sample",
             "",
             f"{len(highlight)} presentation-gene entries (GSTT1/TPMT/CYP2D6/GSTM1) "
             f"+ {len(sample) - len(highlight)} random entries.", ""]
    for variant, gene, adr in sample:
        lines.append(f"## {variant} ({gene}) / {adr}")
        lines.append("")
        lines.append(narrative_cache[_cache_key(variant, adr)])
        lines.append("")

    with open(QA_SAMPLE_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {len(sample)} entries to {QA_SAMPLE_PATH}")


# ── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=[
        "fetch-gene-functions", "draft-adr-pathways", "synthesize", "apply", "qa-sample",
    ])
    args = parser.parse_args()
    {
        "fetch-gene-functions": cmd_fetch_gene_functions,
        "draft-adr-pathways": cmd_draft_adr_pathways,
        "synthesize": cmd_synthesize,
        "apply": cmd_apply,
        "qa-sample": cmd_qa_sample,
    }[args.command]()


if __name__ == "__main__":
    main()
