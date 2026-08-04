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

Run this whenever nodes.json changes (i.e. after pulling a KG rebuild) —
run_alliance.bash does this automatically by comparing file mtimes.
"""
import json
import os

from openai import OpenAI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NODES_PATH = os.path.join(SCRIPT_DIR, "OncologyKG", "kg_export", "nodes.json")
OUT_PATH = os.path.join(SCRIPT_DIR, "OncologyKG", "kg_export", "entity_embeddings.json")

EMBED_LABELS = ["Gene", "Drug", "ADR", "Variant"]
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
EMBED_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")


def main():
    with open(NODES_PATH, encoding="utf-8") as f:
        nodes = json.load(f)

    names = sorted({
        n["properties"]["name"]
        for n in nodes
        if n["label"] in EMBED_LABELS
    })
    print(f"Embedding {len(names):,} entity names ({', '.join(EMBED_LABELS)}) "
          f"with {EMBED_MODEL}...", flush=True)

    client = OpenAI(base_url=EMBED_BASE_URL, api_key="ollama")
    records = []
    for i, name in enumerate(names, 1):
        resp = client.embeddings.create(model=EMBED_MODEL, input=name)
        records.append({"name": name, "embedding": resp.data[0].embedding})
        if i % 50 == 0 or i == len(names):
            print(f"  {i}/{len(names)}", flush=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f)
    print(f"Wrote {len(records):,} embeddings to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
