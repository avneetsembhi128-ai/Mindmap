"""
Precomputes an embedding vector for every Gene/Drug/ADR/Variant entity name in
the OncologyKG graph, so OncologyKGMM.py can do live cosine-similarity matching
(the original MindMap's entity-resolution method) instead of exact-string
lookups. Reads straight from OncologyKG/kg_export/nodes.json (portable, no
live Neo4j connection needed) and writes entity_embeddings.json next to it.

Variant is included (not just Gene/Drug/ADR) so that when a patient names a
specific variant (e.g. "CYP2D6*4"), matching can resolve to that exact node
instead of only ever being able to reach its parent gene — without Variant in
the pool, a named variant has no way to be the actual match, confirmed by
checking real similarity scores: "CYP2D6*4" matched "CYP2D6" at 0.92 simply
because no CYP2D6*4 node was a candidate at all, not because CYP2D6 was truly
the best match available.

The text sent to the embedding model (embed_text) is enriched with each
node's own KG properties — e.g. a bare rsID like "rs10981694" becomes
"rs10981694 (SLC31A1, FKBP15)" using its gene_symbols property, and gene
symbols like "RRM1" get their full_name appended. This is purely about
giving the model more semantic signal to embed; the "name" field written to
entity_embeddings.json stays the exact canonical KG name, unchanged, since
OncologyKGMM.py matches back to nodes by that exact string.

Run this whenever nodes.json changes (i.e. after pulling a KG rebuild) —
run_alliance.bash does this automatically by comparing file mtimes.
"""

import json
import os
import re
from openai import OpenAI

# Allows path to work regardless of where script is run from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Input - Allows script to run without database Neo4j connection
NODES_PATH = os.path.join(SCRIPT_DIR, "OncologyKG", "kg_export", "nodes.json")
# Output - Where computed embeddings get written, next to input file
OUT_PATH = os.path.join(SCRIPT_DIR, "OncologyKG", "kg_export", "entity_embeddings.json")

# Only the gene, drug, adr, and variant labels get embedded.
EMBED_LABELS = ["Gene", "Drug", "ADR", "Variant"]
# Which embedding model to use - reads from environment label if set else uses nomic-embed-text
# nomic-embed-text is a embedding model that runs on Ollama locally
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
EMBED_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")

# Fallback for variants whose gene_symbols are empty but have hgvs string 
# HGVS example "NM_000367.5(TPMT):c.420-4G>A" captures "TPMT"
HGVS_GENE_RE = re.compile(r"\(([A-Za-z0-9\-]+)\):")


def _clean(value):
    """Blank out missing/placeholder property values (None, "", "nan")."""
    text = str(value).strip() if value is not None else ""
    return "" if text.lower() == "nan" else text

# Decides what text gets embedded per node type. 
def build_embed_text(label, props):
    """Text actually sent to the embedding model — enriched with whatever
    semantic context this node's own properties already carry, so a name
    that's just an opaque ID (rsID, gene symbol) isn't embedded bare."""
    name = props["name"]

    if label == "Gene":
        full_name = _clean(props.get("full_name"))
        if full_name and full_name.lower() != name.lower():
            return f"{name} ({full_name})"
        return name

    if label == "Drug":
        drug_class = _clean(props.get("drug_class"))
        if drug_class:
            return f"{name} ({drug_class})"
        return name

    if label == "ADR":
        # ctcae_term/source_terms are true one-to-one synonyms for ADR nodes
        # (e.g. Hepatotoxicity / "Drug-induced liver injury") — safe to fold
        # in. NB: this is not true for Variant's source_terms, see below.
        extras = []
        for key in ("ctcae_term", "source_terms"):
            val = _clean(props.get(key))
            if val and val.lower() != name.lower() and val not in extras:
                extras.append(val)
        if extras:
            return f"{name} ({'; '.join(extras)})"
        return name

    if label == "Variant":
        # NOT using source_terms here: for star-allele variants
        # it's a list of sibling alleles (e.g. UGT1A1*1's source_terms is
        # "UGT1A1*1, UGT1A1*28"), not synonyms of this node — folding it in
        # would blur *1 and *28 together, the same failure shape as the
        # gene_symbols first()-wins bug, just inside the embedding text.
        gene_symbols = _clean(props.get("gene_symbols"))
        if not gene_symbols:
            hgvs = props.get("hgvs")
            match = HGVS_GENE_RE.search(hgvs) if hgvs else None
            gene_symbols = match.group(1) if match else ""
        if gene_symbols and gene_symbols.lower() not in name.lower():
            genes = ", ".join(g.strip() for g in gene_symbols.split(","))
            return f"{name} ({genes})"
        return name

    return name


def main():
    # Load entire node export into memory as a Python list of dicts.
    with open(NODES_PATH, encoding="utf-8") as f:
        nodes = json.load(f)

    # Keep (label, properties) for every embeddable node, deduplicated by
    # (label, name) — first occurrence wins — and sorted by name for stable,
    # diffable output. Neo4j's own name-uniqueness constraint (see
    # load_into_neo4j in kg.py) means this shouldn't ever trigger on a
    # nodes.json produced by this repo's own export, but it's cheap
    # insurance against a hand-edited or pre-constraint export.
    entities_by_key = {}
    for n in nodes:
        if n["label"] in EMBED_LABELS:
            key = (n["label"], n["properties"]["name"])
            entities_by_key.setdefault(key, (n["label"], n["properties"]))
    entities = sorted(entities_by_key.values(), key=lambda entity: entity[1]["name"])

    # Progress message before starting embedding loop
    print(f"Embedding {len(entities):,} entity names ({', '.join(EMBED_LABELS)}) "
          f"with {EMBED_MODEL}...", flush=True)

    # Create client that talks to Ollama server via OpenAI SDK interface.
    client = OpenAI(base_url=EMBED_BASE_URL, api_key="ollama")

    # will hold {"name": ..., "label": ..., "embed_text": ..., "embedding": [...]}
    # dicts, one per entity
    records = []

    # Numbering starts from 1 for progress printout
    for i, (label, props) in enumerate(entities, 1):
        name = props["name"]
        embed_text = build_embed_text(label, props)
        # embedding call - sends the enriched text to Ollama and gets a vector representation of its meaning
        resp = client.embeddings.create(model=EMBED_MODEL, input=embed_text)
        records.append({
            "name": name,
            "label": label,
            "embed_text": embed_text,
            "embedding": resp.data[0].embedding,
        })
        # print progress every 50 entities
        if i % 50 == 0 or i == len(entities):
            print(f"  {i}/{len(entities)}", flush=True)

    # Writes every record to disk as a JSON array
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f)
    print(f"Wrote {len(records):,} embeddings to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
