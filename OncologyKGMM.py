import os
import sys
import re
import csv
import json
import itertools
from time import sleep
import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from openai import OpenAI

# Reuses enrich_mechanisms.py's prompt-building and on-disk caches
# (gene_function_cache.json / adr_pathway_cache.json / mechanism_narrative_cache.json)
# so on-demand synthesis below produces identical output to the batch pipeline,
# without duplicating that logic here.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "OncologyKG"))
import enrich_mechanisms as em
import kg  # canonical Drug names (kg.TARGET_DRUGS), for validating Output2 citation targets

# ==========================================
# 1. API CONNECTION CONFIGURATION
# ==========================================
BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
HEADERS = {}
MODEL = os.environ.get("LLM_MODEL", "qwen3:8b")
# Limit how much neighbor evidence is retrieved from KG - prevents long prompts sent to LLM.
NEIGHBOR_CAP_PER_ENTITY = 15
MAX_NEIGHBOR_EVIDENCE = 30

# Entity-matching embedding model (see generate_entity_embeddings.py) — original
# MindMap's fuzzy cosine-similarity matching, adapted to embed keywords live at
# runtime instead of a precomputed closed-vocabulary lookup (see conversation for why).
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
EMBED_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")

# Minimum cosine similarity to accept a match at all, rather than forcing a
# best-available-but-wrong pick. Calibrated against real observed scores: strong
# correct matches score 0.6-0.92+, while genuinely ambiguous/low-confidence
# matches (e.g. "severe shortness of breath" -> Hypersensitivity 0.50 vs
# Cardiotoxicity 0.49, an effective coin flip) score around 0.50. This does NOT
# catch confidently-wrong matches where the wrong candidate outscores the right
# one (e.g. "Platinol" -> paclitaxel 0.63 > cisplatin 0.61) — no threshold can
# fix that, since both scores look equally confident.
MIN_MATCH_SIMILARITY = 0.55

# NEW: relation-type -> human-readable label, applied in Python (get_entity_neighbors)
# instead of asking the LLM to translate raw relation type names itself. Doing this in
# code closes a real failure mode: on a small local model, "translate this string using
# a 5-way lookup table" is unreliable under the load of everything else the prompt is
# asking for — the model echoed raw types like "ASSOCIATED WITH ADR" verbatim instead of
# the intended phrase in testing. Baking the label into the evidence text itself means
# the LLM's job becomes "copy what's already here," which is far more reliable.
REL_TYPE_LABELS = {
    "LINKED_TO_ADR": "Direct link to this ADR (PharmGKB clinical variant evidence)",
    "ASSOCIATED_WITH_ADR": "Direct link to this ADR (single-study literature annotation)",
    "CLINVAR_ASSOCIATED_ADR": "Direct link to this ADR (ClinVar clinical significance)",
    "AFFECTS_RESPONSE_TO": "Affects response to this drug (PharmGKB clinical variant evidence)",
    "PHARMACOGENOMIC_ASSOCIATION": "Affects response to this drug (literature-annotated mechanism)",
    "HAS_SIDE_EFFECT": "General drug-ADR association (no genetic variant involved)",
}

# Create the OpenAI compatible client to communicate with Ollama.
client = OpenAI(
    base_url=BASE_URL,
    api_key="ollama",
    default_headers=HEADERS,
)

# ==========================================
# 2. CORE INFERENCE FUNCTIONS (NATIVE OPENAI) 
# This section has : Keyword extraction, KG Translation, Evidence Synthesis and Reasoning, and Retry Logic and Guardrails. 
# ==========================================

# Simple general purpose chat completion helper - calls openAI with user prompt and returns response
def chat_35(prompt):
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in chat_35: {e}", flush=True)
        return ""

# Extracts relevant entities from patient text. Entities are matched to the actual
# KG vocabulary downstream via embedding similarity, not here — this step just pulls
# out what the text mentions, in its own words.
def prompt_extract_keyword(input_text):
    prompt = f"""You are a pharmacogenomics assistant. Extract entities from the following patient text.

ONLY extract: drug names (including brand names, exactly as written), gene/variant names if
mentioned (exactly as written), and adverse drug reaction (ADR) or symptom terms (in the patient's
own words, do not rename or categorize them).

Input Text: {input_text}

Output the extracted entities as a clean, comma-separated list. Do not include introductory text.
"""
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error extracting keywords: {e}", flush=True)
        return ""


# Translates raw KG path relationship to readable sentence
def prompt_path_finding(path_input):
    prompt = f"""There are some knowledge graph paths. They follow entity->relationship->entity format.
Some relationships carry a bracketed [mechanism: ...; evidence: "..."; confidence: ...] annotation —
that is real data from the knowledge graph explaining WHY the relationship holds, not commentary.

{path_input}

Use the knowledge graph information. Convert them to natural language sentences cleanly. Use single
quotation marks for entity names and relation names. Name them as Path-based Evidence 1, Path-based
Evidence 2, etc. Whenever a relationship has a [mechanism: ...] or [evidence: "..."] annotation,
explicitly state that mechanism/evidence text as part of the sentence — this is the answer to "why"
and must not be dropped.

State only what each path literally says — do not add mechanisms, severity, or clinical explanation
that isn't present in the path/annotations themselves. Do not omit any path.

Output:"""
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Network hitch on path finding, retrying... Error: {e}", flush=True)
        sleep(2)
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content

# Translates raw KG neighbor connections into readable sentences. 
def prompt_neighbor(neighbor):
    prompt = f"""There are some knowledge graph connections. They follow entity->relationship->entity list format.
Some connections carry a bracketed [mechanism: ...; evidence: "..."; confidence: ...] annotation —
that is real data from the knowledge graph explaining WHY the connection holds, not commentary.

{neighbor}

Use the knowledge graph information. Convert them to natural language sentences cleanly. Use single
quotation marks for entity names and relation names. Name them as Neighbor-based Evidence 1,
Neighbor-based Evidence 2, etc. Whenever a connection has a [mechanism: ...] or [evidence: "..."]
annotation, explicitly state that mechanism/evidence text as part of the sentence — this is the
answer to "why" and must not be dropped.

State only what each connection literally says — do not add mechanisms, severity, or clinical
interpretation that isn't present in the data/annotations themselves. If a connection lists multiple
neighbor entities, mention all of them.

Output:"""
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Network hitch on neighbor prompting, retrying... Error: {e}", flush=True)
        sleep(2)
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content

# Main reasoning that synthesizes inputs, graph path evidence, neighbor evidence,
# and pre-written (grounded) mechanism narratives into the final answer.
def final_answer(question_text, response_of_KG_list_path, response_of_KG_neighbor, mechanism_narratives):
    prompt = f"""You are an excellent clinical pharmacogenomicist, and you explain how genetic variants affect drug metabolism and cause adverse drug reactions (ADRs), based on the patient case in the conversation.

Patient input: {question_text}

You have some pharmacogenomic knowledge information retrieved from a knowledge graph, in the following:
### {response_of_KG_list_path}
### {response_of_KG_neighbor}

Pre-written mechanism narratives (already grounded in real gene-function data, generated separately
ahead of time — wherever one of these matches an entity below, reuse it near-verbatim, do not
rewrite, compress, or replace it with your own explanation):
### {mechanism_narratives}

What genetic variants or genes could explain the patient's adverse drug reaction? What is the
mechanism — WHY would this variant actually cause the reaction, not just that it's associated with
it? Think step by step.

"The evidence above" means all three blocks together — the knowledge-graph path/neighbor summaries
AND the pre-written mechanism narratives block — not just the first two. An entity named ONLY in the
pre-written mechanism narratives block still counts as found and must be covered; that block is
built directly from real, verified graph queries, so it is not less trustworthy than the path/
neighbor summaries just because it's presented separately. Never present a gene/variant as if it
came from the evidence above unless it is actually named in at least one of the three blocks. If all
three blocks are empty or name no specific gene/variant, say so plainly rather than filling the gap
silently — but if the mechanism narratives block is non-empty, it is never correct to say nothing
was found.

Output1: For EACH gene/variant/phenotype explicitly named in the evidence above (see definition
just given — this includes anything named in the pre-written mechanism narratives block), give one
block in this exact format — no bullets, no "Relationship:"/"Evidence:" labels, no sub-headers, just
the name followed directly by its paragraph:

[Gene/Variant/Phenotype name]
[A single flowing paragraph of 2-4 sentences explaining WHY this entity would cause or contribute
to the ADR/drug response — not merely that it's associated with it. Follow a causal chain: what the
variant/genotype does to the gene's normal function -> the resulting biochemical consequence -> how
that leads to the ADR. Style example: "Looking at the GSTT1 null gene variant genotype, where the
gene is deleted. This leads to reduced production of GSTT1 enzyme, which would normally help
neutralize harmful byproducts of cisplatin. Since there is less of that enzyme around, the toxic
metabolites don't get removed as efficiently and stay around longer, which leads to the higher risk
of hearing loss."]

If a pre-written mechanism narrative above names this same entity (and, where relevant, the same
ADR), that near-verbatim narrative IS the paragraph — light grammatical smoothing only, never
compress it, never drop a step of its causal chain, never substitute your own explanation instead of
it. If there is no pre-written narrative for this entity, write the paragraph yourself, in the same
causal-chain style, grounded in the KG evidence for this entity (its relationship to the drug/ADR)
plus your own general pharmacology knowledge of the gene's function — it must not contradict what
the evidence above says about this entity. Stay tightly focused on explaining THIS entity's link to
THIS patient's drug/ADR — do not pad the paragraph with the gene's other substrates, other drugs it
affects, or general biology trivia that isn't needed for this specific causal chain.

Each entity's paragraph must be about THAT entity only. Never mention, compare against, or hedge
using a different gene/variant inside this entity's paragraph — if another entity is also relevant,
it gets its own separate block, not a mention inside this one. If the only KG evidence for this
entity concerns a different drug than the one in the patient's question, or gives no explanatory
description at all (just a bare association), say that plainly in one clause and keep the rest of
the paragraph to what the gene's real function actually supports — do not invent a specific
mechanism connecting it to the patient's drug/ADR that the evidence doesn't establish, and do not
turn the paragraph into a hedging comparison with another entity instead. Once you state that a
mechanism isn't established, STOP — do not add a further sentence speculating how it might still be
connected anyway ("may alter...", "could lead to...", "potentially..."). Hedged speculation is still
fabrication, just softened.

Every heading in Output1 must be copied EXACTLY as the entity's real name appears in the evidence
above — never invent a generic or combined label (e.g. do not write "[TPMT variant]" to cover both
TPMT*1 and TPMT*3A; give each its own separate block, "[TPMT*1]" and "[TPMT*3A]", using their real
names). If Output2 or the evidence names two distinct entities, Output1 must have two distinct
entries, never one merged one.

Do not create an entry for the drug or the ADR themselves, or for any structural gene-variant
relationship that isn't drug/ADR evidence — only genuine drug- or ADR-linked findings belong here.
Output2's citation lines follow the same rule: only cite genes/variants/phenotypes that have their
own Output1 entry — never cite the drug or ADR itself as if it were a finding, even for a general
drug-ADR association with no gene involved (that case is handled by the fallback sentence below, not
a citation line).
If no gene/variant/phenotype is named in the evidence at all, skip Output1's entries and say so
plainly in one sentence: "No specific gene or variant was identified in the knowledge graph or
general knowledge for this case."

If the ONLY evidence found is a general drug-ADR association with no gene/variant/phenotype
involved at all, do NOT create any entries — instead state plainly: "The knowledge graph confirms
[drug] is a known cause of [ADR] in general, but does not identify a specific genetic variant
responsible."

Output2 (technical evidence trail, for verification): For each gene/variant/phenotype in Output1,
give ONE line in this exact format: "[citation label]: [Gene/Variant name] -> [relationship] -> [ADR
or Drug]". The citation label already says which type it is (e.g. "Path-based Evidence 1" or
"Neighbor-based Evidence 18") — do not add a separate "(Path-based/Neighbor-based)" tag, and do not
split this into two separate headed sections. List every line together as ONE combined sequence, in
whatever order they naturally come up — mixing Path-based and Neighbor-based lines interleaved is
correct and expected, matching how the citation label itself already identifies the type. Copy the
citation label exactly as given in the evidence above — never renumber, never reuse a label, never
invent one. Never cite the same underlying fact
under two or more different citation labels, even if it looks similar — if you would write the same
chain twice, only include it once. Never combine multiple hops into one narrative sentence (e.g. "X
shows A->B, which connects to C->D") — one line per fact only, no chains, no connecting language. A
line about one gene/variant must mention ONLY that gene/variant — never name a different
gene/variant/allele (such as a comparison partner from the source sentence) inside another entity's
line. If a fact is supported by both a Path-based and a Neighbor-based citation, use the
Path-based one only, since it reflects a more specific traversal. Never construct a
'Patient'->'has'->... chain or any structure not literally present in the evidence.

If a gene/variant/phenotype from Output1 doesn't appear in either evidence block by its citation
label, omit it rather than guessing which one it came from. If Output1 found no KG-grounded
evidence at all, write exactly: "No evidence-based inference chain available."

Output3 (technical evidence trail, for verification): Draw exactly ONE ASCII tree diagram with a
SINGLE root node — never produce more than one separate tree, even if multiple keywords (drug, ADR,
gene/variant, disease context) were extracted from the patient's question. Use the primary matched
entity as the single root (the drug if one was matched, otherwise the ADR). Every other extracted
keyword, and every gene/variant/phenotype from Output1, must be nested as a branch somewhere within
this ONE tree, showing how it connects to the root — not drawn as its own separate top-level tree.
Build this only from Output1 items — every word in this tree must be traceable to Output1's
paragraph for that entity. Do NOT add any mechanism, explanation, or descriptive phrase that isn't
already written in Output1, even if it sounds plausible — a tree branch is a short label summarizing
Output1's paragraph, not a new sentence.

Before finalizing, check: does the tree contain any gene/variant/phenotype, or any wording, that
Output1 didn't already state? If so, remove it. Does the tree include both real branches AND the
sentence "No KG-grounded decision tree available"? These two outcomes are mutually exclusive and
must never appear together — check for this specifically before finishing:
- If Output1 lists ANY gene/variant/phenotype, draw the full single tree with every one of them
  included as a branch, using ONLY labels traceable to Output1, and do NOT write "No KG-grounded
  decision tree available" anywhere in Output3.
- If Output1 found NO KG-grounded evidence at all (the entries were skipped), show only the root
  keyword with no further branches, and write "No KG-grounded decision tree available" underneath —
  in this case only, do not draw any branches at all.

Final consistency check across all three outputs — do this before finalizing, as a last pass over
what you've written: Output1, Output2, and Output3 must all name the exact same set of
gene/variant/phenotype entities — no more, no fewer. Concretely: list out every entity that has a
paragraph in Output1; every one of them must also have at least one citation line in Output2 and a
branch in Output3; and neither Output2 nor Output3 may contain any entity that does NOT have a
paragraph in Output1. If you find a mismatch — an entity cited in Output2 or drawn in Output3 without
an Output1 paragraph, or an Output1 entity missing from Output2/Output3 — fix it by adding the
missing piece or removing the extra one. Never submit an answer where the three outputs disagree
about which entities are covered.

Follow this exact structure style for your reply:
Output 1:
[Gene/Variant/Phenotype 1]
[Flowing 2-4 sentence causal-chain paragraph]

[Gene/Variant/Phenotype 2]
[Flowing 2-4 sentence causal-chain paragraph]

Output 2:
[citation label]: [Gene/Variant name] -> [relationship] -> [ADR or Drug]
[citation label]: [Gene/Variant name] -> [relationship] -> [ADR or Drug]

Output 3:
[extracted keywords]
└── [relationship]
    └── [Gene/Variant/Phenotype]
"""
    # This response covers Output1+2+3 for potentially many entities (9+ seen
    # in testing) — no max_tokens/truncation protection existed here at all
    # until now, and truncated text (cut off mid-sentence, e.g. "...there is
    # no established textbook pathway linking this function to") was
    # observed reaching the final displayed output. Same fix as the
    # narrative-synthesis calls: explicit generous budget + retry once on
    # finish_reason == "length" rather than silently keeping a cut-off answer.
    try:
        for attempt in range(2):
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4000,
            )
            choice = completion.choices[0]
            if choice.finish_reason != "length":
                return choice.message.content
            print(f"  (final_answer truncated at finish_reason=length, retrying, attempt {attempt + 1})", flush=True)
        return choice.message.content  # both attempts truncated — return what we have
    except Exception as e:
        print(f"Network hitch on final answer generation, retrying... Error: {e}", flush=True)
        sleep(2)
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
        )
        return completion.choices[0].message.content

# Alternative single-document reasoning prompt.
def prompt_document(question, instruction):
    prompt = f"""You are an excellent clinical pharmacogenomicist, and you explain adverse drug reactions based on the patient case in the conversation.

Patient input:
{question}

You have some pharmacogenomic knowledge information in the following:
{instruction}

What genetic variants or genes could explain the patient's adverse drug reaction?"""
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in prompt_document: {e}", flush=True)
        return ""

# Checks if LLM response waas successful to the user. 
def is_unable_to_answer(response_text):
    judge_prompt = (
        "You are evaluating whether a response actually answers the question, "
        "or instead avoids, declines, or fails to answer it.\n\n"
        f'Response to evaluate:\n"""\n{response_text}\n"""\n\n'
        "Rate how well this response answers the question, from 0.0 "
        '(completely fails to answer / refuses / says "I don\'t know") '
        "to 1.0 (fully and directly answers).\n"
        "Reply with ONLY the number, nothing else."
    )
    try:
        analysis = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=3,
            temperature=0.0,
        )
        raw = analysis.choices[0].message.content.strip()
        match = re.search(r"\d*\.?\d+", raw)
        if not match:
            return True
        score = float(match.group())
        return score <= 0.6
    except:
        return False


# ==========================================
# 3. GRAPH HELPER FUNCTIONS
# Section covers - combinatorial utility, shortest path traversal, and bidirectional neighborhood retrieval. 
# ==========================================

# Generates all possible combinations across multiple entity lists
def combine_lists(*lists):
    # FIX: itertools.product() called with ZERO arguments doesn't return an
    # empty result — it returns one empty tuple ([()]), its documented
    # "identity" behavior. That flowed through into a single-item list
    # containing one EMPTY path, so len(result_path) came out as 1 even when
    # truly zero paths were found (logged as "Found 1 paths." with
    # "direction: none"). That phantom path then went into
    # prompt_path_finding("") instead of leaving response_of_KG_list_path as
    # the literal "{}" the rest of the pipeline checks for — silently
    # bypassing the safeguard against fabricated Path-based Evidence
    # citations exactly when it mattered most (the true-zero-path case).
    if not lists:
        return []
    combinations = list(itertools.product(*lists))
    results = []
    for combination in combinations:
        new_combination = []
        for sublist in combination:
            if isinstance(sublist, list):
                new_combination += sublist
            else:
                new_combination.append(sublist)
        results.append(new_combination)
    return results

# Parse Cypher path objects returned by Neo4j into readable string, keeping each
# hop's description/mechanism/confidence so path evidence explains "why" too.
def _parse_path_records(records, candidate_list):
    global exist_entity
    paths = []
    short_path = 0
    for record in records:
        path = record["p"]
        entities = []
        relations = []
        contexts = []
        for i in range(len(path.nodes)):
            node = path.nodes[i]
            entity_name = node["name"]
            entities.append(entity_name)
            if i < len(path.relationships):
                relationship = path.relationships[i]
                relations.append(relationship.type)
                contexts.append(_format_edge_context(dict(relationship.items())))

        path_str = ""
        for i in range(len(entities)):
            entities[i] = entities[i].replace("_", " ")
            # candidate_list holds match_kg's underscored form; entities[i] is
            # the real (spaced) DB name — normalize to underscored form for
            # this comparison only, same root cause as find_shortest_path's fix.
            if entities[i].replace(" ", "_") in candidate_list:
                short_path = 1
                exist_entity = entities[i]
            path_str += entities[i]
            if i < len(relations):
                relations[i] = relations[i].replace("_", " ")
                rel_obj = path.relationships[i]
                true_start_name = rel_obj.start_node["name"].replace("_", " ")
                if true_start_name == entities[i]:
                    path_str += "->" + relations[i] + contexts[i] + "->"
                else:
                    path_str += "<-" + relations[i] + contexts[i] + "<-"

        if short_path == 1:
            paths = [path_str]
            break
        else:
            paths.append(path_str)
            exist_entity = {}

    if len(paths) > 5:
        paths = sorted(paths, key=len)[:5]
    return paths, exist_entity

# Find graph paths connecting two entities with fallback logic. Entity names
# arrive here in match_kg's underscored form (e.g. "Peripheral_Neuropathy"),
# but real Neo4j node names use spaces ("Peripheral Neuropathy") — querying
# with the underscored form directly matches zero nodes for any multi-word
# entity, silently producing "0 paths found" regardless of what's actually in
# the graph. Convert to the real DB form before querying; candidate_list stays
# in its original (underscored) form since it's compared against match_kg
# elsewhere, so the comparison in _parse_path_records normalizes instead.
def find_shortest_path(start_entity_name, end_entity_name, candidate_list):
    global exist_entity
    start_db_name = start_entity_name.replace("_", " ")
    end_db_name = end_entity_name.replace("_", " ")
    with driver.session() as session:
        result = session.run(
            "MATCH (start_entity{name:$start_entity_name}), (end_entity{name:$end_entity_name}) "
            "MATCH p = allShortestPaths((start_entity)-[*..5]->(end_entity)) "
            "RETURN p",
            start_entity_name=start_db_name,
            end_entity_name=end_db_name
        )
        records = list(result)

    paths, exist_entity = _parse_path_records(records, candidate_list)
    if paths and paths != ['']:
        return paths, exist_entity, "directed"

    # Directed search found nothing — try again ignoring edge direction.
    with driver.session() as session:
        result = session.run(
            "MATCH (start_entity{name:$start_entity_name}), (end_entity{name:$end_entity_name}) "
            "MATCH p = allShortestPaths((start_entity)-[*..5]-(end_entity)) "
            "RETURN p",
            start_entity_name=start_db_name,
            end_entity_name=end_db_name
        )
        records = list(result)

    paths, exist_entity = _parse_path_records(records, candidate_list)
    if paths and paths != ['']:
        print(f"  Note: path {start_entity_name}->{end_entity_name} found only "
              f"UNDIRECTED — evidence direction in the prompt may not reflect true causality.")
        return paths, exist_entity, "undirected_fallback"

    return paths, exist_entity, "none"

# Formats the "why" context carried on an edge (description/mechanism/confidence
# properties) into a short suffix, or "" if the edge has nothing informative.
# An ADR node's neighbors span EVERY drug that causes it — structurally
# different from a Gene/Variant/Drug node, whose neighbors are already about
# that specific entity. Confirmed against the graph directly: all 22 of
# Peripheral Neuropathy's variant-ADR edges, checked with zero filtering,
# came back with a cisplatin-asking question — none of them actually
# mention cisplatin in their description; every one is about a different
# drug (bortezomib, paclitaxel, oxaliplatin, vincristine, ...). Those
# neighbors are real graph facts, just not relevant to the drug this
# specific question is actually about — get_entity_neighbors filters them
# out when fetching an ADR's neighbors alongside a matched Drug, using this
# to check whether a given neighbor edge is actually about that drug.
def _drug_class_term(drug_name):
    """The drug's class's most distinctive word (e.g. "platinum" from
    kg.TARGET_DRUGS's "Platinum compound" for cisplatin) — PharmGKB source
    data frequently describes a finding by drug CLASS rather than naming the
    specific drug. Confirmed against the graph directly: several genuinely
    cisplatin-relevant SLC31A1 variants (rs4979223, rs4978536, rs10817464,
    rs10759637 — the same variants from the rs10981694/FKBP15 investigation)
    are described only as "platinum compounds" or bare "platinum," never
    literally "cisplatin" — matching only the literal drug name would
    silently treat every one of these as irrelevant. The class's first word
    is used rather than the whole phrase since PharmGKB's wording varies
    ("platinum compound" / "platinum compounds" / bare "platinum") and the
    first word is reliably the distinctive one across this project's drug
    classes (Platinum compound, Anthracycline, Vinca alkaloid, Alkylating
    agent, Topoisomerase inhibitor, ...)."""
    words = re.findall(r"[A-Za-z]+", kg.TARGET_DRUGS.get(drug_name.lower(), ""))
    return words[0].lower() if words else None


def _edge_mentions_drug(neighbor_name, props, drug_name):
    drug_lower = drug_name.lower()
    if neighbor_name.lower() == drug_lower:
        return True
    haystack = f"{props.get('description', '')} {props.get('source_term', '')}".lower()
    if drug_lower in haystack:
        return True
    class_term = _drug_class_term(drug_name)
    return bool(class_term and re.search(rf"\b{re.escape(class_term)}s?\b", haystack))


def _format_edge_context(props):
    parts = []
    mechanism = props.get("mechanism")
    if mechanism and mechanism.lower() != "unknown":
        parts.append(f"mechanism: {mechanism}")
    description = props.get("description")
    if description:
        parts.append(f'evidence: "{description}"')
    confidence = props.get("confidence")
    if confidence:
        parts.append(f"confidence: {confidence}")
    return f" [{'; '.join(parts)}]" if parts else ""


# Cosine similarity between one query vector (y) and a matrix of candidate vectors
# (x, one row per entity) — same formula as the original MindMap's entity-resolution
# logic, corrected for a broadcasting mismatch the original had when y is 1D (it
# produced an [N,N] matrix instead of an [N] vector; taking row [0] of that, as the
# original did, silently used entity 0's norm as the denominator for every entity).
def cosine_similarity_manual(x, y):
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y)
    return dot_product / (norm_x * norm_y)


# Fetch 1-hop neighbors around an entity across both edge directions, including
# each edge's description/mechanism/confidence so the "why" isn't lost, AND a
# human-readable relationship label (see REL_TYPE_LABELS) so the LLM doesn't have
# to translate the raw relation type name itself.
#
# Returns dicts (not plain strings) so callers can both build the formatted
# evidence text AND recover the raw neighbor_name — the latter is what lets
# the main loop check every entity that will actually appear in Output1's
# evidence for a pre-written mechanism narrative, not just the entities
# name-matched directly from the question (see get_mechanism_narratives).
def get_entity_neighbors(entity_name: str, drug_filter: str = None) -> list:
    # entity_name arrives in match_kg's underscored form; real Neo4j node
    # names use spaces — see find_shortest_path's comment for why querying
    # with the underscored form directly silently matches zero nodes for any
    # multi-word entity.
    db_name = entity_name.replace("_", " ")
    outgoing_query = """
    MATCH (e)-[r]->(n)
    WHERE e.name = $entity_name
    RETURN type(r) AS relationship_type, n.name AS neighbor_name, properties(r) AS props
    ORDER BY relationship_type, neighbor_name
    """
    incoming_query = """
    MATCH (n)-[r]->(e)
    WHERE e.name = $entity_name
    RETURN type(r) AS relationship_type, n.name AS neighbor_name, properties(r) AS props
    ORDER BY relationship_type, neighbor_name
    """
    with driver.session() as session:
        neighbor_list = []

        result = session.run(outgoing_query, entity_name=db_name)
        for record in result:
            rel_type = record["relationship_type"]
            rel_label = REL_TYPE_LABELS.get(rel_type, rel_type.replace('_', ' '))
            neighbor_name = record["neighbor_name"]
            props = record["props"]
            if drug_filter and not _edge_mentions_drug(neighbor_name, props, drug_filter):
                continue
            context = _format_edge_context(props)
            neighbor_list.append({
                "neighbor_name": neighbor_name,
                "text": f"{entity_name.replace('_', ' ')}->{rel_label}->"
                        f"{neighbor_name.replace('_', ' ')}{context}",
            })

        result = session.run(incoming_query, entity_name=db_name)
        for record in result:
            rel_type = record["relationship_type"]
            rel_label = REL_TYPE_LABELS.get(rel_type, rel_type.replace('_', ' '))
            neighbor_name = record["neighbor_name"]
            props = record["props"]
            if drug_filter and not _edge_mentions_drug(neighbor_name, props, drug_filter):
                continue
            context = _format_edge_context(props)
            neighbor_list.append({
                "neighbor_name": neighbor_name,
                "text": f"{neighbor_name.replace('_', ' ')}->{rel_label}->"
                        f"{entity_name.replace('_', ' ')}{context}",
            })
    return neighbor_list


# Mechanism narrative lookup for a matched Variant entity — on-demand, not
# batch. Rather than requiring OncologyKG/enrich_mechanisms.py's `synthesize`
# step to have already pre-computed a narrative for every gene/variant in the
# graph (1,500+ tuples, ~8 hours of LLM calls), this fetches/synthesizes a
# narrative ONLY for the entities a given question actually matched — usually
# 1-3 per question. The two things synthesis depends on (each gene's UniProt
# function text, and the 10 ADR pathway paragraphs) are still pre-fetched
# ahead of time by enrich_mechanisms.py's `fetch-gene-functions` and
# `draft-adr-pathways` steps — both cheap, both already complete — so the
# only work happening live here is the per-entity synthesis call itself.
# Results are written back to Neo4j (r.mechanism_narrative) and to the same
# mechanism_narrative_cache.json enrich_mechanisms.py uses, so a gene/variant
# is only ever synthesized once, whether that happens via a batch run or by
# coming up in a real question first.
# PharmGKB's consistent phrasing for a null/negative finding — verified
# against multiple real edges this session, always some variant of
# "not associated"/"no association", never a different phrasing for the
# same concept.
_NULL_FINDING_RE = re.compile(r"\bnot associated\b|\bno association\b", re.IGNORECASE)


def _is_null_finding(description):
    return bool(description) and bool(_NULL_FINDING_RE.search(description))


# Lower tuple sorts first = stronger evidence. PharmGKB evidence_level runs
# "1A" (strongest) through "4" (weakest, single study); confidence is a
# separate low/medium/high tag. Used to rank narratives before they reach
# final_answer, so a well-replicated finding (e.g. a "1A"/high-confidence
# SNP) is presented ahead of a weaker single-study candidate — previously
# narratives were an unordered list and final_answer picked an arbitrary
# subset with no evidence-strength signal at all.
def _evidence_score(confidence, evidence_level):
    conf_rank = {"high": 0, "medium": 1, "low": 2}.get((confidence or "").lower(), 3)
    level_num, level_letter = 9, "Z"  # unknown/missing = worst
    m = re.match(r"(\d+)([A-Za-z]?)", (evidence_level or "").strip())
    if m:
        level_num = int(m.group(1))
        level_letter = m.group(2) or "A"
    return (level_num, level_letter, conf_rank)


def get_mechanism_narratives(entity_name: str, llm_client) -> list:
    with driver.session() as session:
        # HAS_CLINICAL_VARIANT (PharmGKB's clinicalVariants table) is a more
        # specific signal than v.gene_symbols below, which just lists every
        # gene whose genomic span overlaps this variant's coordinate — see
        # enrich_mechanisms.py's _variant_to_gene for the full reasoning
        # (confirmed never ambiguous when present, across the whole graph).
        clinical_genes = {
            row["gene"] for row in session.run(
                "MATCH (g:Gene)-[:HAS_CLINICAL_VARIANT]->(v:Variant {name: $entity_name}) "
                "RETURN DISTINCT g.name AS gene",
                entity_name=entity_name,
            )
        }
        result = session.run(
            """
            MATCH (v:Variant {name: $entity_name})-[r]->(adr:ADR)
            WHERE type(r) IN ['LINKED_TO_ADR', 'ASSOCIATED_WITH_ADR', 'CLINVAR_ASSOCIATED_ADR']
            RETURN adr.name AS adr_name, r.mechanism_narrative AS narrative,
                   r.direction AS direction, r.side_effect_type AS side_effect_type,
                   r.description AS description, v.gene_symbols AS gene_symbols,
                   r.confidence AS confidence, r.evidence_level AS evidence_level
            """,
            entity_name=entity_name,
        )
        rows = list(result)

    if not rows:
        return []

    # Group by ADR, not by edge: a (variant, ADR) pair commonly has more than
    # one edge (e.g. LINKED_TO_ADR from clinicalVariants.tsv AND
    # ASSOCIATED_WITH_ADR from variantAnnotations.tsv) citing the exact same
    # underlying finding — synthesizing one per edge produced duplicate
    # near-identical narratives for what's really one fact.
    by_adr = {}
    for row in rows:
        by_adr.setdefault(row["adr_name"], []).append(row)

    narratives = []
    gene_functions = adr_pathways = narrative_cache = None  # lazy-loaded only if actually needed

    for adr, group in by_adr.items():
        # Skip explicit null/no-association findings entirely — e.g. "Allele C
        # is NOT associated with risk of Ototoxicity... as compared to allele
        # T." Output1 exists to explain WHY a variant contributes to the ADR;
        # a variant the graph explicitly says has no association contributes
        # nothing to explain, and listing it alongside real risk-increasing
        # variants is misleading even when the narrative text itself is
        # honestly worded. Checked before the cache lookup below too, so a
        # narrative already cached from before this filter existed doesn't
        # slip through.
        descriptions = [r["description"] for r in group if r["description"]]
        if descriptions and all(_is_null_finding(d) for d in descriptions):
            continue

        score = min((_evidence_score(r["confidence"], r["evidence_level"]) for r in group), default=(9, "Z", 3))

        existing = next((r["narrative"] for r in group if r["narrative"]), None)
        if existing:
            narratives.append({"entity": entity_name, "adr": adr, "narrative": existing, "score": score})
            continue

        if gene_functions is None:
            gene_symbols = group[0]["gene_symbols"] or ""
            if len(clinical_genes) == 1:
                gene = next(iter(clinical_genes))
            elif gene_symbols and gene_symbols.strip().lower() != "nan":
                gene = gene_symbols.split(",")[0].strip()
            else:
                gene = entity_name.split(" ")[0]
            gene_functions = em._load_json(em.GENE_FUNCTION_CACHE, default={})
            function_text = gene_functions.get(gene)
            if not function_text:
                return narratives  # no grounding for this gene — leave it to Output1's general-knowledge fallback
            adr_pathways = em._load_json(em.ADR_PATHWAY_CACHE, default={})
            narrative_cache = em._load_json(em.NARRATIVE_CACHE, default={})

        direction = next((r["direction"] or r["side_effect_type"] for r in group if (r["direction"] or r["side_effect_type"])), "")
        description = next((r["description"] for r in group if r["description"]), "")
        key = em._cache_key(entity_name, adr)

        text = narrative_cache.get(key)
        if not text:
            prompt = em.build_narrative_prompt(
                gene, function_text, adr, adr_pathways.get(adr, ""),
                direction, [entity_name], description,
            )
            text = em._generate_narrative(llm_client, prompt, model=MODEL)
            narrative_cache[key] = text
            em._save_json(em.NARRATIVE_CACHE, narrative_cache)
            print(f"  Synthesized on demand: '{entity_name}' / {adr}", flush=True)

        with driver.session() as session:
            session.run(
                "MATCH (v:Variant {name:$name})-[r]->(adr:ADR {name:$adr}) "
                "WHERE type(r) IN ['LINKED_TO_ADR', 'ASSOCIATED_WITH_ADR', 'CLINVAR_ASSOCIATED_ADR'] "
                "SET r.mechanism_narrative = $text",
                name=entity_name, adr=adr, text=text,
            )
        narratives.append({"entity": entity_name, "adr": adr, "narrative": text, "score": score})

    return narratives


def format_mechanism_narratives(narratives: list) -> str:
    if not narratives:
        return "(none — no pre-written narrative available for any matched entity)"
    lines = ["(listed strongest evidence first — prefer entries earlier in this list "
             "if you have to choose which ones to feature)"]
    for n in narratives:
        lines.append(
            f"- Entity: '{n['entity'].replace('_', ' ')}' | ADR: '{n['adr']}' | "
            f"Narrative: {n['narrative']}"
        )
    return "\n".join(lines)


NO_ENTITY_FALLBACK = ("No specific gene or variant was identified in the knowledge graph or "
                       "general knowledge for this case.")
NO_EVIDENCE_CHAIN_FALLBACK = "No evidence-based inference chain available."
NO_DECISION_TREE_FALLBACK = "No KG-grounded decision tree available"


# Raw-generation cleanup, run BEFORE any of the Output1/2/3 parsing below —
# every function past this point assumes exactly one clean "Output 1: ...
# Output 2: ... Output 3: ..." sequence, line-scanning for the first match of
# each header. Two artifacts observed in testing break that assumption:
#
# 1. qwen3 thinking-mode leakage: a bare, dangling "/think" line with no
#    matching open tag (not a full <think>...</think> block — that never
#    appeared; just this fragment), left sitting after Output 3.
# 2. A fully duplicated answer: two complete Output1/2/3 sequences back to
#    back for the same question, the second copy visibly degraded (a
#    corrupted entity name, "doxorubic,icin", and its own "/think" leakage).
#    Keeping both confuses every downstream function, which would find the
#    SECOND "Output 2:"/"Output 3:" as easily as the first — only the first
#    complete sequence is real evidence of anything.
_THINK_LEAK_RE = re.compile(r"^\s*/?think\s*$", re.IGNORECASE | re.MULTILINE)


def strip_thinking_and_duplicates(output_text: str) -> str:
    text = _THINK_LEAK_RE.sub("", output_text)

    lines = text.split("\n")
    out1_positions = [i for i, l in enumerate(lines) if re.match(r"\s*Output\s*1\s*:", l)]
    if len(out1_positions) > 1:
        print(f"  Duplicate answer detected ({len(out1_positions)} 'Output 1:' blocks in one "
              f"response) — keeping only the first.", flush=True)
        lines = lines[:out1_positions[1]]
    return "\n".join(lines).strip()


# Deterministic fix for a persistent failure mode: even after the prompt was
# rewritten multiple times to say pre-written narratives count as evidence,
# and even after filtering the narratives block down to just the matched
# ADR(s), final_answer has still been observed writing the "nothing found"
# fallback while real, relevant narratives were sitting right there in the
# prompt (confirmed: 10 Peripheral-Neuropathy-specific narratives available,
# Output1 still claimed nothing was found). Rather than trying yet another
# prompt wording, construct Output1/2/3 directly from the narratives we KNOW
# are real whenever this specific contradiction is detected — narratives
# non-empty, but Output1 claims none exist.
def fix_wrongly_empty_output(output_text: str, narratives: list) -> str:
    if not narratives:
        return output_text

    lines = output_text.split("\n")
    out1_idx = next((i for i, l in enumerate(lines) if re.match(r"\s*Output\s*1\s*:", l)), None)
    out2_idx = next((i for i, l in enumerate(lines) if re.match(r"\s*Output\s*2\s*:", l)), None)
    out3_idx = next((i for i, l in enumerate(lines) if re.match(r"\s*Output\s*3\s*:", l)), None)
    if out1_idx is None or out2_idx is None or out3_idx is None or not (out1_idx < out2_idx < out3_idx):
        return output_text

    output1_body = "\n".join(lines[out1_idx + 1:out2_idx])
    if NO_ENTITY_FALLBACK not in output1_body:
        return output_text  # Output1 already has real content — leave everything alone

    print("  Output1 wrongly claimed nothing was found despite real narratives — "
          "constructing it directly instead.", flush=True)

    seen = set()
    output1_blocks, output2_lines, by_adr = [], [], {}
    for n in narratives:
        if n["entity"] in seen:
            continue
        seen.add(n["entity"])
        output1_blocks.append(f"[{n['entity']}]\n{n['narrative']}")
        output2_lines.append(
            f"[Mechanism-narrative Evidence]: {n['entity']} -> Direct link to this ADR -> {n['adr']}"
        )
        by_adr.setdefault(n["adr"], []).append(n["entity"])

    output3_lines = []
    for adr, entities in by_adr.items():
        output3_lines.append(f"[{adr}]")
        output3_lines.append("└── [Direct link to this ADR]")
        for j, entity in enumerate(entities):
            branch = "└──" if j == len(entities) - 1 else "├──"
            output3_lines.append(f"    {branch} [{entity}]")

    new_lines = (
        lines[:out1_idx + 1]
        + [""] + ["\n\n".join(output1_blocks)] + [""]
        + lines[out2_idx:out2_idx + 1]
        + [""] + output2_lines + [""]
        + lines[out3_idx:out3_idx + 1]
        + [""] + output3_lines
    )
    return "\n".join(new_lines)


# Deterministic enforcement of pre-written narrative fidelity. final_answer is
# instructed to reproduce a pre-written narrative "near-verbatim... never
# substitute your own explanation instead of it" — but tested against a real
# run, this is NOT reliably followed: the exact same cached, honest
# "GSTM1 null / Cardiotoxicity" narrative (which correctly states the
# mechanism is unestablished) got silently replaced with three DIFFERENT
# fabricated "established mechanism" versions across three different
# questions in the same run, and only matched the real cached text on the
# 4th appearance. Trusting the LLM to carry text through its own generation
# unchanged is the same failure mode that motivated fetching narratives
# directly from Neo4j in the first place (see get_mechanism_narratives) —
# same fix applies here: stop trusting it, guarantee it in Python instead.
def _extract_entity_adr_map(output_text: str) -> dict:
    """Reads Output2's own citation lines to determine which ADR each cited
    entity is actually about IN THIS SPECIFIC ANSWER. More reliable than
    inferring it from match_kg upstream — many questions never get an ADR
    into match_kg at all (e.g. one that only matches the drug name), so an
    upstream ADR filter has nothing to filter by, even though Output2's own
    citations always end in a real target entity."""
    entity_adr = {}
    canonical_lower = {a.lower(): a for a in em.CANONICAL_ADRS}
    lines = output_text.split("\n")
    start = next((i for i, l in enumerate(lines) if re.match(r"\s*Output\s*2\s*:", l)), None)
    end = next((i for i, l in enumerate(lines) if re.match(r"\s*Output\s*3\s*:", l)), None)
    if start is None or end is None:
        return entity_adr
    for line in lines[start:end]:
        if ":" not in line or "->" not in line:
            continue
        fact = line.split(":", 1)[1]
        parts = [p.strip().strip("[]").strip() for p in fact.split("->")]
        if len(parts) < 2:
            continue
        entity, target = parts[0], parts[-1]
        if target.lower() in canonical_lower:
            entity_adr[entity] = canonical_lower[target.lower()]
    return entity_adr


def enforce_prewritten_narratives(output_text: str, narratives: list) -> str:
    if not narratives:
        return output_text

    # Keyed by (entity, adr) primarily — an entity with cached narratives for
    # several different ADRs (e.g. GSTM1 null has both a Cardiotoxicity one
    # and an Ototoxicity one) must get the one matching what THIS answer is
    # actually about, not just whichever happens to be first in the list.
    # Confirmed in testing: without ADR-awareness, a Cardiotoxicity-question
    # narrative got silently injected into an unrelated Ototoxicity question.
    by_entity_adr = {}
    by_entity_fallback = {}
    for n in narratives:
        by_entity_adr.setdefault((n["entity"], n["adr"]), n["narrative"])
        by_entity_fallback.setdefault(n["entity"], n["narrative"])

    entity_adr_map = _extract_entity_adr_map(output_text)

    lines = output_text.split("\n")
    out1_idx = next((i for i, l in enumerate(lines) if re.match(r"\s*Output\s*1\s*:", l)), None)
    out2_idx = next((i for i, l in enumerate(lines) if re.match(r"\s*Output\s*2\s*:", l)), None)
    if out1_idx is None or out2_idx is None or out2_idx <= out1_idx:
        return output_text  # format not as expected — leave untouched rather than risk mangling it

    new_lines = lines[:out1_idx + 1]
    i = out1_idx + 1
    while i < out2_idx:
        heading = lines[i].strip().strip("[]").strip()
        adr = entity_adr_map.get(heading)
        narrative = by_entity_adr.get((heading, adr)) if adr else None
        if narrative is None:
            narrative = by_entity_fallback.get(heading)
        if narrative is not None:
            new_lines.append(lines[i])  # keep the heading as the model wrote it
            new_lines.append(narrative)  # force the real, cached, ADR-matched narrative
            i += 1
            while i < out2_idx and lines[i].strip() != "":
                i += 1  # skip whatever paragraph the model wrote instead
        else:
            new_lines.append(lines[i])
            i += 1
    new_lines += lines[out2_idx:]
    return "\n".join(new_lines)


# Deterministic dedup of Output1 itself: observed the model writing the exact
# same entity (heading + full paragraph) as two separate top-level blocks in
# the same answer (e.g. "rs10981694" appearing twice, back to back, with
# identical text both times). Keep only the first occurrence of each heading.
def dedupe_output1_entities(output_text: str) -> str:
    lines = output_text.split("\n")
    out1_idx = next((i for i, l in enumerate(lines) if re.match(r"\s*Output\s*1\s*:", l)), None)
    out2_idx = next((i for i, l in enumerate(lines) if re.match(r"\s*Output\s*2\s*:", l)), None)
    if out1_idx is None or out2_idx is None or out2_idx <= out1_idx:
        return output_text

    new_lines = lines[:out1_idx + 1]
    seen = set()
    i = out1_idx + 1
    while i < out2_idx:
        if lines[i].strip() == "":
            # Pass blank separator lines through untouched — without this,
            # the next iteration reads the blank line itself as a "heading"
            # (empty string), which never matches `seen` and desyncs block
            # boundaries for every heading after the first.
            new_lines.append(lines[i])
            i += 1
            continue
        heading = lines[i].strip().strip("[]").strip()
        block_start = i
        i += 1
        while i < out2_idx and lines[i].strip() != "":
            i += 1
        if heading in seen:
            continue  # drop this whole repeated block (heading + paragraph)
        seen.add(heading)
        new_lines.extend(lines[block_start:i])
    new_lines += lines[out2_idx:]
    return "\n".join(new_lines)


# Deterministic filter of Output2 down to entities that actually have an
# Output1 paragraph. The consistency instruction in the prompt already asks
# for this, but adherence keeps slipping (e.g. citing "GSTT1 null",
# "TPMT*3B + *3C" in Output2 with zero corresponding Output1 entries) — same
# guarantee-it-in-code approach as everything else here.
def filter_output2_to_output1_entities(output_text: str) -> str:
    lines = output_text.split("\n")
    out1_idx = next((i for i, l in enumerate(lines) if re.match(r"\s*Output\s*1\s*:", l)), None)
    out2_idx = next((i for i, l in enumerate(lines) if re.match(r"\s*Output\s*2\s*:", l)), None)
    out3_idx = next((i for i, l in enumerate(lines) if re.match(r"\s*Output\s*3\s*:", l)), None)
    if out1_idx is None or out2_idx is None or out3_idx is None or not (out1_idx < out2_idx < out3_idx):
        return output_text

    output1_entities = set()
    for line in lines[out1_idx + 1:out2_idx]:
        heading = line.strip().strip("[]").strip()
        if heading:
            output1_entities.add(heading)
    if not output1_entities:
        return output_text  # no real entities in Output1 — leave Output2's fallback text alone

    # Structural Gene<->Variant relations (see kg.py) carry no ADR/drug
    # evidence at all — REL_TYPE_LABELS has no human-readable translation for
    # them, so they surface with their raw type name (e.g. "HAS CLINICAL
    # VARIANT"), which is itself a signal they were never meant to be cited
    # as evidence. Observed one reaching Output2: "rs2291767 -> HAS CLINICAL
    # VARIANT -> OTOS" — a real edge, but not evidence of anything ADR-related.
    STRUCTURAL_RELATIONS = {"HAS CLINICAL VARIANT", "HAS VARIANT", "IN GENE"}

    # Relation/target type-consistency check. Confirmed against the graph
    # directly: every real AFFECTS_RESPONSE_TO edge points at a Drug node,
    # zero exceptions — so a citation like "GSTM1 null -> affects response to
    # this drug -> Cardiotoxicity" (an ADR name where a drug belongs) isn't a
    # KG data problem, it's final_answer mislabeling or merging two separate
    # real edges when writing the citation. Drop citations where the
    # relation's implied target type doesn't match what the target actually
    # is — an ADR-labeled relation whose target is a real Drug name, or vice
    # versa. Anything whose target isn't a recognized canonical ADR/Drug name
    # (e.g. another Gene/Variant, for structural or comparison edges) is left
    # alone rather than guessed at.
    ADR_RELATION_PHRASES = ("direct link to this adr", "linked to adr", "associated with adr")
    DRUG_RELATION_PHRASES = ("affects response to", "pharmacogenomic association", "pharmacogenomic associated")
    canonical_adrs_lower = {a.lower() for a in em.CANONICAL_ADRS}

    def _target_type(name):
        name_lower = name.lower()
        if name_lower in canonical_adrs_lower:
            return "adr"
        if name_lower in kg.TARGET_DRUGS:
            return "drug"
        return None

    new_block = []
    for line in lines[out2_idx:out3_idx]:
        if ":" in line and "->" in line:
            fact = line.split(":", 1)[1]
            parts = [p.strip().strip("[]").strip() for p in fact.split("->")]
            entity = parts[0] if parts else ""
            if entity not in output1_entities:
                continue  # cites something Output1 never actually discusses — drop it
            relation = parts[1].upper() if len(parts) >= 2 else ""
            if relation in STRUCTURAL_RELATIONS:
                continue  # structural edge, not real ADR/drug evidence — drop it
            if len(parts) >= 2:
                relation_lower = parts[1].lower()
                target_type = _target_type(parts[-1])
                implies_adr = any(p in relation_lower for p in ADR_RELATION_PHRASES)
                implies_drug = any(p in relation_lower for p in DRUG_RELATION_PHRASES)
                if (implies_adr and target_type == "drug") or (implies_drug and target_type == "adr"):
                    continue  # relation type and actual target disagree — drop it
        new_block.append(line)
    return "\n".join(lines[:out2_idx] + new_block + lines[out3_idx:])


# Deterministic cleanup of Output2's citation lines: even though the prompt
# already instructs the model not to cite the same fact under two labels
# (e.g. both "Path-based Evidence 1" and "Neighbor-based Evidence 27" citing
# the identical "GSTM1 null -> ASSOCIATED WITH ADR -> Cardiotoxicity"), a
# small local model doesn't always follow that reliably. Checked against the
# actual graph in cases where this happened — there was only ONE real edge
# each time, so the duplication is a generation artifact, not duplicate KG
# data. Fixing it here in Python guarantees it rather than hoping the prompt
# is followed, since path-finding and neighbor-fetching are independent
# traversals that can genuinely rediscover the same edge.
def dedupe_output2_citations(output_text: str) -> str:
    lines = output_text.split("\n")
    start = next((i for i, l in enumerate(lines) if re.match(r"\s*Output\s*2\s*:", l)), None)
    end = next((i for i, l in enumerate(lines) if re.match(r"\s*Output\s*3\s*:", l)), None)
    if start is None or end is None or end <= start:
        return output_text  # format not as expected — leave untouched rather than risk mangling it

    seen_facts = set()
    deduped_block = []
    for line in lines[start:end]:
        if ":" in line and "->" in line:
            fact = line.split(":", 1)[1].strip().lower()
            fact = re.sub(r"\s+", " ", fact)
            if fact in seen_facts:
                continue  # drop this line — same fact already cited under a different label
            seen_facts.add(fact)
        deduped_block.append(line)

    return "\n".join(lines[:start] + deduped_block + lines[end:])


# Extracts just the real entity headings from an Output1 span — the first
# non-blank line of each blank-line-delimited block, matching how
# dedupe_output1_entities above walks blocks. NOT the same as the simpler
# "every non-blank line" scan filter_output2_to_output1_entities uses above:
# that shortcut only ever gets used for `in` membership checks against short
# citation labels, where an extra bogus "entity" (a full paragraph sentence)
# sitting unused in the set is harmless. The two functions below actually
# enumerate and act on every entity found, so a paragraph sentence
# misidentified as a heading would get its own fabricated citation/branch —
# confirmed by testing this exact scan against real multi-entity output
# before the fix. Also excludes NO_ENTITY_FALLBACK itself: when Output1 has
# no real entities, that fallback sentence IS the first (only) non-blank
# line, and without this exclusion it gets treated as a pseudo-entity too —
# confirmed by testing the no-entity case before adding this filter.
def _output1_headings(lines: list, out1_idx: int, out2_idx: int) -> list:
    headings = []
    i = out1_idx + 1
    while i < out2_idx:
        if lines[i].strip() == "":
            i += 1
            continue
        heading = lines[i].strip().strip("[]").strip()
        if heading != NO_ENTITY_FALLBACK:
            headings.append(heading)
        i += 1
        while i < out2_idx and lines[i].strip() != "":
            i += 1  # skip the rest of this block's paragraph lines
    return headings


# Deterministic completeness guarantee for Output1 itself — nothing before
# this checks whether Output1 actually contains a paragraph for every entity
# the pipeline found real, grounded evidence for. Confirmed missing in
# testing: a question whose mechanism_narrative_list (after ADR-scoping) had
# ~10 on-topic, cached-narrative-backed entities — including "rs1695",
# genuinely relevant via a real cisplatin/Ototoxicity finding, discovered
# through path-traversal rather than neighbor-fetching (see the
# "Path-discovered entities" loop in the main pipeline below) — produced an
# Output1 with only 2 of them. Every other guarantee in this file
# (ensure_output2/3_covers_output1_entities) only enforces consistency
# between whatever Output1 happens to contain and Output2/3; none of them
# check whether Output1 itself is complete relative to the real evidence it
# was handed. Same fix as everywhere else: for any narrative-list entity
# still missing after generation, append its real cached paragraph directly
# (same block-construction style as fix_wrongly_empty_output) rather than
# trust the model to have included it. Runs BEFORE the Output2/3 coverage
# passes below so anything appended here also gets a citation/branch.
def ensure_output1_covers_narratives(output_text: str, narratives: list) -> str:
    if not narratives:
        return output_text

    lines = output_text.split("\n")
    out1_idx = next((i for i, l in enumerate(lines) if re.match(r"\s*Output\s*1\s*:", l)), None)
    out2_idx = next((i for i, l in enumerate(lines) if re.match(r"\s*Output\s*2\s*:", l)), None)
    if out1_idx is None or out2_idx is None or out2_idx <= out1_idx:
        return output_text  # format not as expected — leave untouched rather than risk mangling it

    existing_headings = set(_output1_headings(lines, out1_idx, out2_idx))

    seen, missing_blocks = set(), []
    for n in narratives:
        if n["entity"] in existing_headings or n["entity"] in seen:
            continue
        seen.add(n["entity"])
        missing_blocks.append(f"[{n['entity']}]\n{n['narrative']}")

    if not missing_blocks:
        return output_text

    print(f"  Output1 missing {len(missing_blocks)} grounded narrative "
          f"entit{'y' if len(missing_blocks) == 1 else 'ies'} ({', '.join(seen)}) — appending.", flush=True)

    body = lines[out1_idx + 1:out2_idx]
    if len(body) == 1 and body[0].strip() == NO_ENTITY_FALLBACK:
        body = []  # replacing the "nothing found" fallback with real entries
    while body and body[-1].strip() == "":
        body.pop()

    return "\n".join(
        lines[:out1_idx + 1] + body + [""] + ["\n\n".join(missing_blocks)] + [""] + lines[out2_idx:]
    )


# Deterministic enforcement of the OTHER direction of coverage:
# filter_output2_to_output1_entities above only ever removes citations that
# Output1 doesn't back — nothing guaranteed the reverse. Confirmed missing in
# testing on two separate real answers: "GSTT1 null" had a full Output1
# paragraph (built from a real cached mechanism narrative) but zero Output2
# citation, and "CYP2D6*4" had the same gap in a different question — even
# though both were backed by a real, structured mechanism_narratives entry
# the whole time (verified against mechanism_narrative_cache.json directly).
# Same guarantee-it-in-code approach as everything else here: for any
# Output1 entity still missing from Output2 after the passes above, if a
# cached narrative names it, synthesize the same real citation format
# fix_wrongly_empty_output already uses elsewhere; if no structured source
# names it at all, add a line that's honest about that gap rather than
# fabricate a fake Path-based/Neighbor-based label pointing at nothing.
def ensure_output2_covers_output1_entities(output_text: str, narratives: list) -> str:
    lines = output_text.split("\n")
    out1_idx = next((i for i, l in enumerate(lines) if re.match(r"\s*Output\s*1\s*:", l)), None)
    out2_idx = next((i for i, l in enumerate(lines) if re.match(r"\s*Output\s*2\s*:", l)), None)
    out3_idx = next((i for i, l in enumerate(lines) if re.match(r"\s*Output\s*3\s*:", l)), None)
    if out1_idx is None or out2_idx is None or out3_idx is None or not (out1_idx < out2_idx < out3_idx):
        return output_text

    output1_entities = _output1_headings(lines, out1_idx, out2_idx)
    if not output1_entities:
        return output_text

    cited_entities = set()
    for line in lines[out2_idx:out3_idx]:
        if ":" in line and "->" in line:
            fact = line.split(":", 1)[1]
            parts = [p.strip().strip("[]").strip() for p in fact.split("->")]
            if parts:
                cited_entities.add(parts[0])

    missing = [e for e in output1_entities if e not in cited_entities]
    if not missing:
        return output_text

    narrative_adr_by_entity = {}
    for n in narratives:
        narrative_adr_by_entity.setdefault(n["entity"], n["adr"])

    print(f"  Output1 entities missing from Output2 ({', '.join(missing)}) — adding citations.", flush=True)
    added_lines = []
    for entity in missing:
        adr = narrative_adr_by_entity.get(entity)
        if adr:
            added_lines.append(f"[Mechanism-narrative Evidence]: {entity} -> Direct link to this ADR -> {adr}")
        else:
            added_lines.append(f"{entity}: discussed in Output1 — no separate structured KG citation captured for this entity")

    out2_body = [l for l in lines[out2_idx + 1:out3_idx] if l.strip() != NO_EVIDENCE_CHAIN_FALLBACK]
    return "\n".join(lines[:out2_idx + 1] + out2_body + added_lines + lines[out3_idx:])


# Same guarantee, applied to Output3's tree. Confirmed in the same "CYP2D6*4"
# case above: it was entirely absent from Output3's tree even though the
# other two Output1 entities in that same answer were both present as
# branches. Rather than trying to re-derive the "correct" nesting position
# in the tree (fragile — the model's tree shape/root choice varies question
# to question, sometimes rooted at the drug, sometimes the ADR), any Output1
# entity missing from Output3 is appended as its own branch — guaranteeing
# it's visible rather than silently dropped.
#
# The appended branch is nested at the SAME indentation as the tree's own
# top-level branch (matched from the first "├──"/"└──" line found), not left
# at column 0 — a bare "└── [entity]" with no indentation reads as a second,
# disconnected top-level tree rather than a branch of the real one (confirmed
# visually: it renders after the whole tree with nothing showing it belongs
# to the same root). If the current last top-level branch uses "└──" (the
# "I'm the final child" marker), it's converted to "├──" first, since it's
# no longer the last child once a new sibling is appended after it — keeps
# the ASCII tree-art connector convention valid instead of having two lines
# both claim to be the last one.
_TREE_BRANCH_RE = re.compile(r"^(\s*)(├──|└──)")


def ensure_output3_covers_output1_entities(output_text: str) -> str:
    lines = output_text.split("\n")
    out1_idx = next((i for i, l in enumerate(lines) if re.match(r"\s*Output\s*1\s*:", l)), None)
    out2_idx = next((i for i, l in enumerate(lines) if re.match(r"\s*Output\s*2\s*:", l)), None)
    out3_idx = next((i for i, l in enumerate(lines) if re.match(r"\s*Output\s*3\s*:", l)), None)
    if out1_idx is None or out2_idx is None or out3_idx is None or not (out1_idx < out2_idx < out3_idx):
        return output_text

    output1_entities = _output1_headings(lines, out1_idx, out2_idx)
    if not output1_entities:
        return output_text

    tree_lines = lines[out3_idx + 1:]
    tree_text = "\n".join(tree_lines)
    missing = [e for e in output1_entities if e not in tree_text]
    if not missing:
        return output_text

    print(f"  Output1 entities missing from Output3's tree ({', '.join(missing)}) — appending branches.", flush=True)
    tree_lines = [l for l in tree_lines if NO_DECISION_TREE_FALLBACK not in l]

    top_level_indent = ""
    for l in tree_lines:
        m = _TREE_BRANCH_RE.match(l)
        if m:
            top_level_indent = m.group(1)
            break

    for i in range(len(tree_lines) - 1, -1, -1):
        m = _TREE_BRANCH_RE.match(tree_lines[i])
        if m and m.group(1) == top_level_indent:
            if m.group(2) == "└──":
                tree_lines[i] = tree_lines[i].replace("└──", "├──", 1)
            break

    for entity in missing:
        tree_lines.append(f"{top_level_indent}├── [{entity}]")
    tree_lines[-1] = tree_lines[-1].replace("├──", "└──", 1)  # true last child

    return "\n".join(lines[:out3_idx + 1] + tree_lines)


# Output2's citation lines (Path-based/Neighbor-based/Mechanism-narrative,
# regardless of type) should run contiguously — a blank line means "next
# section," not "next citation type." The model's own natural output already
# does this correctly (confirmed: real Path-based and Neighbor-based lines
# always come out back to back with no gap between them); the gap this fixes
# comes from ensure_output2_covers_output1_entities above, which preserves
# whatever blank line originally sat right before "Output 3:" but then
# appends its new citation lines AFTER that blank — landing the one gap
# between old and new citations instead of before the next header. Run this
# last so it cleans up regardless of which upstream step introduced it.
def normalize_output2_spacing(output_text: str) -> str:
    lines = output_text.split("\n")
    out2_idx = next((i for i, l in enumerate(lines) if re.match(r"\s*Output\s*2\s*:", l)), None)
    out3_idx = next((i for i, l in enumerate(lines) if re.match(r"\s*Output\s*3\s*:", l)), None)
    if out2_idx is None or out3_idx is None or out3_idx <= out2_idx:
        return output_text

    body = [l for l in lines[out2_idx + 1:out3_idx] if l.strip() != ""]
    return "\n".join(lines[:out2_idx + 1] + body + [""] + lines[out3_idx:])


# ==========================================
# 4. MAIN PIPELINE EXECUTION
# ==========================================
if __name__ == "__main__":

    # Environment and Neo4j connection setup 
    NEO4J_URI      = os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687")
    NEO4J_USER     = os.environ.get("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
    if not NEO4J_PASSWORD:
        raise SystemExit(
            "Set the NEO4J_PASSWORD environment variable before running this script "
            "(must match the password used to build/load the graph in OncologyKG).\n"
            "PowerShell:  $env:NEO4J_PASSWORD = \"your-password-here\"\n"
            "bash:        export NEO4J_PASSWORD=\"your-password-here\""
        )
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Precomputed KG-entity embeddings (see generate_entity_embeddings.py) — the
    # "entity_embeddings.pkl" side of the original MindMap's matching method.
    embeddings_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "OncologyKG", "kg_export", "entity_embeddings.json"
    )
    with open(embeddings_path, encoding="utf-8") as f:
        entity_embedding_records = json.load(f)
    entity_names = [r["name"] for r in entity_embedding_records]
    entity_embeddings_matrix = np.array([r["embedding"] for r in entity_embedding_records])

    embed_client = OpenAI(base_url=EMBED_BASE_URL, api_key="ollama")

    # CSV output initialization and test questions
    with open('output.csv', 'w', newline='', encoding='utf-8') as f4:
        writer = csv.writer(f4)
        writer.writerow(['Question', 'MindMap'])
    TEST_QUESTIONS = [
        "I am a child receiving cisplatin for cancer and developed hearing loss and tinnitus. What genetic variants make me susceptible to this ototoxicity?",
        "What genes or variants have been linked to cisplatin ototoxicity risk in pediatric cancer patients?",
        "My daughter carries a TPMT variant and is on cisplatin for her Neuroblastoma. Her audiologist says she's high risk for hearing loss — is there a genetic reason, and does her cancer type change the picture?",
        "After several cycles of Platinol, my child has permanent inner ear damage, and we're wondering if her GSTT1 status played a role, since her oncologist mentioned it during a different conversation about drug metabolism.",
        "Can cisplatin cause a genetic predisposition to peripheral neuropathy?",

        "After receiving doxorubicin for pediatric lymphoma, my child developed severe shortness of breath and heart muscle damage. Is there a genetic cause for this?",
        "Which pharmacogenomic biomarkers predict anthracycline-induced cardiotoxicity in childhood cancer survivors?",
        "My son carries CYP2D6*4 and was on doxorubicin for his Neoplasm — his cardiologist is concerned about early heart failure. Is this genetic, and does his specific cancer diagnosis matter here?",
        "My daughter took Adriamycin and now has cardiac strain and heart muscle damage — could her GSTM1 result from an earlier test be connected?",
        "Could doxorubicin cause a genetic predisposition to mucositis?",
    ]

    for input_text_0 in TEST_QUESTIONS:
        print('\nQuestion:\n', input_text_0, flush=True)

        # Dynamic entity extraction and database lookup
        raw_entities = prompt_extract_keyword(input_text_0)
        print('Raw extracted entities:', raw_entities, flush=True)
        question_kg = [e.strip() for e in raw_entities.split(",") if e.strip()]

        if not question_kg:
            print("<Warning> no entities found", input_text_0, flush=True)
            continue

        # Entity matching via live-embedded cosine similarity — the original
        # MindMap's method (see conversation: same algorithm, keyword side embedded
        # at runtime instead of a precomputed closed-vocabulary lookup). A keyword
        # is skipped entirely (not force-matched) if even its best candidate falls
        # below MIN_MATCH_SIMILARITY — see that constant's comment for why.
        match_kg = []
        for kg_entity in question_kg:
            kg_entity_resp = embed_client.embeddings.create(model=EMBED_MODEL, input=kg_entity)
            kg_entity_emb = np.array(kg_entity_resp.data[0].embedding)

            cos_similarities = cosine_similarity_manual(entity_embeddings_matrix, kg_entity_emb)
            max_index = cos_similarities.argmax()
            match_kg_i = entity_names[max_index]

            while match_kg_i.replace(" ", "_") in match_kg:
                cos_similarities[max_index] = 0
                max_index = cos_similarities.argmax()
                match_kg_i = entity_names[max_index]

            if cos_similarities[max_index] < MIN_MATCH_SIMILARITY:
                print(f"  Note: '{kg_entity}' best candidate was '{match_kg_i}' at "
                      f"similarity {cos_similarities[max_index]:.3f}, below "
                      f"MIN_MATCH_SIMILARITY={MIN_MATCH_SIMILARITY} — skipping, not matching.",
                      flush=True)
                continue

            match_kg.append(match_kg_i.replace(" ", "_"))

        print('Matched Entities from KG Lookup:', match_kg, flush=True)

        # Multi entity path finding and traversal
        result_path = []
        path_directions_seen = set()
        if len(match_kg) > 1:
            start_entity = match_kg[0]
            candidate_entity = match_kg[1:]
            result_path_list = []

            while True:
                flag = 0
                paths_list = []
                while candidate_entity:
                    end_entity = candidate_entity[0]
                    candidate_entity.remove(end_entity)
                    paths, exist_entity, direction = find_shortest_path(start_entity, end_entity, candidate_entity)
                    path_directions_seen.add(direction)
                    path_list = []
                    if not paths or paths == ['']:
                        flag = 1
                        if not candidate_entity:
                            flag = 0
                            break
                        start_entity = candidate_entity[0]
                        candidate_entity.remove(start_entity)
                        break
                    else:
                        for p in paths:
                            path_list.append(p.split('->'))
                        if path_list:
                            paths_list.append(path_list)

                    if exist_entity != {}:
                        try:
                            candidate_entity.remove(exist_entity)
                        except:
                            pass
                    start_entity = end_entity

                result_path = combine_lists(*paths_list)
                if result_path:
                    result_path_list.extend(result_path)
                if flag == 1:
                    continue
                else:
                    break

            start_tmp = []
            for path_new in result_path_list:
                if not path_new:
                    continue
                if path_new[0] not in start_tmp:
                    start_tmp.append(path_new[0])

            if len(start_tmp) == 1:
                result_path = result_path_list[:3]
            elif len(start_tmp) > 1:
                result_path = result_path_list[:3]

        print(f"Path Finding Complete. Found {len(result_path)} paths. "
              f"(direction: {', '.join(sorted(path_directions_seen)) or 'n/a'})", flush=True)

        # Per entity neighbor fetching. Mechanism narratives are checked for
        # every entity that will actually appear in the evidence text shown to
        # the LLM — the matched entities themselves AND their capped neighbors
        # (e.g. a gene like GSTT1 reached only via being one of Ototoxicity's
        # neighbors, never directly name-matched from the question) — not just
        # match_kg, otherwise entities discovered purely through graph
        # traversal would fall back to an ungrounded narrative for no reason.
        neighbor_list = []
        mechanism_narrative_list = []
        narrative_checked = set()

        def _maybe_fetch_narrative(entity_name):
            if entity_name in narrative_checked:
                return
            narrative_checked.add(entity_name)
            mechanism_narrative_list.extend(get_mechanism_narratives(entity_name, client))

        # An ADR node's neighbors span every drug that causes it (see
        # _edge_mentions_drug) — when this question also matched a specific
        # Drug, scope the ADR's own neighbor fetch to that drug so entities
        # tied only to other drugs don't ride along as if they were relevant
        # evidence for this patient's actual drug. Not applied to Gene/
        # Variant/Drug entities — their neighbors are already about that
        # specific entity, and an off-drug finding there is itself relevant
        # context (e.g. "this gene's real function has nothing to do with
        # the patient's drug" is a useful thing to state, not noise).
        matched_drug = next(
            (e.replace("_", " ") for e in match_kg if e.replace("_", " ").lower() in kg.TARGET_DRUGS),
            None,
        )

        for match_entity in match_kg:
            entity_name = match_entity.replace("_", " ")
            is_adr = entity_name in em.CANONICAL_ADRS
            drug_filter = matched_drug if (is_adr and matched_drug) else None
            neighbors = get_entity_neighbors(match_entity, drug_filter=drug_filter)
            capped = neighbors[:NEIGHBOR_CAP_PER_ENTITY]
            neighbor_list.extend(item["text"] for item in capped)

            _maybe_fetch_narrative(entity_name)
            for item in capped:
                _maybe_fetch_narrative(item["neighbor_name"].replace("_", " "))

        # Path-discovered entities (e.g. TPMT*1/TPMT*3A reached only via the
        # cisplatin<->TPMT path traversal, never as a 1-hop neighbor of any
        # matched entity) were being skipped by the neighbor-only coverage
        # above, forcing final_answer to write their paragraph live with no
        # grounding — which is exactly where the honesty rules kept failing
        # (both TPMT*1 and TPMT*3A got confident fabricated mechanisms in
        # testing, unlike every entity that went through get_mechanism_narratives).
        # Path entities alternate [entity, relation+context, entity, ...] —
        # real entities are at the even indices.
        for path in result_path[:3]:
            for i in range(0, len(path), 2):
                _maybe_fetch_narrative(path[i].replace("_", " "))

        print(f"Neighbor Fetch Complete. Found {len(neighbor_list)} neighbors.", flush=True)
        print(f"Mechanism Narrative Fetch Complete. Found {len(mechanism_narrative_list)} "
              f"pre-written narratives.", flush=True)

        response_of_KG_list_path = "{}"

        # Path summarization prompt
        if result_path and isinstance(result_path, list):
            result_new_path = ["->".join(total_path_i) for total_path_i in result_path[:3]]
            path_str_block = "\n".join(result_new_path)
            response_of_KG_list_path = prompt_path_finding(path_str_block)
            if is_unable_to_answer(response_of_KG_list_path):
                response_of_KG_list_path = prompt_path_finding(path_str_block)

        neighbor_new_list = neighbor_list
        neighbor_input = "\n".join(neighbor_new_list[:MAX_NEIGHBOR_EVIDENCE])
        response_of_KG_neighbor = "{}"

        # Neighbor summarization prompt
        if neighbor_new_list:
            response_of_KG_neighbor = prompt_neighbor(neighbor_input)
            if is_unable_to_answer(response_of_KG_neighbor):
                response_of_KG_neighbor = prompt_neighbor(neighbor_input)

        # Filter to only the ADR(s) this question actually matched. Without this,
        # a variant that's relevant to Peripheral Neuropathy also drags in its
        # narratives for Cardiotoxicity, Mucositis, Hepatotoxicity, etc. (every ADR
        # that variant happens to have an edge to) — pure noise for this question,
        # and at high volume (e.g. 50 narratives spanning 9 ADRs) it appears to
        # overwhelm the model's ability to follow the "treat this block as
        # evidence" instruction, producing "no gene/variant identified" despite
        # real narratives being available. Matching against em.CANONICAL_ADRS
        # (the same 10-ADR list used throughout the KG) identifies which matched
        # entities are actually ADR nodes.
        matched_adrs = {e.replace("_", " ") for e in match_kg if e.replace("_", " ") in em.CANONICAL_ADRS}
        if matched_adrs:
            mechanism_narrative_list = [n for n in mechanism_narrative_list if n["adr"] in matched_adrs]

        # Present strongest evidence first (see _evidence_score) — previously
        # an unordered list, so final_answer had no signal for which of many
        # candidates to feature and could just as easily pick a weak
        # single-study SNP over a well-replicated one.
        mechanism_narrative_list.sort(key=lambda n: n.get("score", (9, "Z", 3)))

        mechanism_narratives_block = format_mechanism_narratives(mechanism_narrative_list)

        output_all = final_answer(input_text_0, response_of_KG_list_path, response_of_KG_neighbor,
                                   mechanism_narratives_block)

        # Final mindmap synthesis
        if is_unable_to_answer(output_all):
            output_all = final_answer(input_text_0, response_of_KG_list_path, response_of_KG_neighbor,
                                       mechanism_narratives_block)

        output_all = strip_thinking_and_duplicates(output_all)
        output_all = fix_wrongly_empty_output(output_all, mechanism_narrative_list)
        output_all = enforce_prewritten_narratives(output_all, mechanism_narrative_list)
        output_all = dedupe_output1_entities(output_all)
        output_all = ensure_output1_covers_narratives(output_all, mechanism_narrative_list)
        output_all = filter_output2_to_output1_entities(output_all)
        output_all = dedupe_output2_citations(output_all)
        output_all = ensure_output2_covers_output1_entities(output_all, mechanism_narrative_list)
        output_all = ensure_output3_covers_output1_entities(output_all)
        output_all = normalize_output2_spacing(output_all)

        print('\nMindMap Output:\n', output_all, flush=True)

        # Results output logging 
        with open('output.csv', 'a+', newline='', encoding='utf-8') as f6:
            writer = csv.writer(f6)
            writer.writerow([input_text_0, output_all])
            f6.flush()

    driver.close()