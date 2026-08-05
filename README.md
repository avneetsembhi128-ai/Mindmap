# MindMap for Pediatric Oncology ADR

This is an adaptation of **MindMap** — [Knowledge Graph Prompting Sparks Graph of
Thoughts in Large Language Models](https://arxiv.org/pdf/2308.09729.pdf) (Wen et al.,
ACL'24) — retargeted to synthesize published research on pediatric oncology adverse
drug reactions (ADRs), using a purpose-built knowledge graph instead of the paper's
original chatdoctor5k dataset.

## How it fits together

```
OncologyKG/                     builds and hosts the knowledge graph (Neo4j)
  kg.py                           build / load / export / audit — see OncologyKG/README.md
  enrich_mechanisms.py            (optional) pre-generates "why" mechanism narratives —
                                   see OncologyKG/README.md
  kg_export/                      committed graph snapshot (nodes.json, edges.json) plus
                                   generated caches (entity_embeddings.json and
                                   enrich_mechanisms.py's caches)

generate_entity_embeddings.py   precomputes an embedding per Gene/Drug/Variant/ADR name,
                                 used for entity-resolution matching below

OncologyKGMM.py                 MindMap's graph-of-thoughts pipeline, adapted to query
                                 OncologyKG instead of chatdoctor5k
```

`OncologyKGMM.py` extracts entities from a clinical question with a local LLM,
looks them up in the Gene/Drug/Variant/ADR/Phenotype graph (via cosine-similarity
matching against `generate_entity_embeddings.py`'s precomputed vectors), finds
paths and neighbor evidence between matched entities, and asks the LLM to
synthesize a grounded answer — the same reasoning flow as the original MindMap,
just pointed at this graph. If a matched Variant/ADR pair has no pre-generated
mechanism narrative (see `enrich_mechanisms.py` in
[OncologyKG/README.md](OncologyKG/README.md)), it synthesizes one on the spot
and caches it, so pre-running that pipeline is optional but avoids repeated
on-demand LLM calls at query time.

On the Digital Research Alliance of Canada cluster this project was developed
on, `run_alliance.bash` runs the whole thing end to end as one SLURM job
(Ollama + local Neo4j + `kg.py load` + entity embeddings if stale + `OncologyKGMM.py`)
— `sbatch run_alliance.bash`. The steps below are the general, portable path.

## Run it

1. **Build/load the graph.** See [OncologyKG/README.md](OncologyKG/README.md) —
   fastest path is `cd OncologyKG && python kg.py load`, which needs a running
   Neo4j instance and `NEO4J_PASSWORD` set. Remember the password you use here —
   step 4 below must match it.

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Have an LLM endpoint available.** By default both `OncologyKGMM.py` and
   `generate_entity_embeddings.py` talk to a local Ollama server — the former
   using the chat model `qwen3:8b`, the latter using the embedding model
   `nomic-embed-text`. Pull both first (`ollama pull qwen3:8b`,
   `ollama pull nomic-embed-text`), or point at different OpenAI-compatible
   endpoints via env vars (see below) instead of editing either script.

4. **Set environment variables** (same convention as `OncologyKG/kg.py` — nothing
   is hardcoded, so this reproduces on another machine without touching source):

   | Variable | Default | Purpose |
   |---|---|---|
   | `NEO4J_PASSWORD` | *(required, no default)* | Must match whatever password you used for `kg.py load`/`build` in step 1 |
   | `NEO4J_URI` | `neo4j://127.0.0.1:7687` | Neo4j connection URI |
   | `NEO4J_USER` | `neo4j` | Neo4j username |
   | `LLM_BASE_URL` | `http://127.0.0.1:11434/v1` | OpenAI-compatible endpoint — chat completions for `OncologyKGMM.py`/`enrich_mechanisms.py`, embeddings for `generate_entity_embeddings.py` |
   | `LLM_MODEL` | `qwen3:8b` | Chat model name, used by `OncologyKGMM.py` and `enrich_mechanisms.py` |
   | `EMBED_MODEL` | `nomic-embed-text` | Embedding model name, used by `generate_entity_embeddings.py` |

   ```bash
   # PowerShell
   $env:NEO4J_PASSWORD = "your-password-here"
   # bash
   export NEO4J_PASSWORD="your-password-here"
   ```

   The script exits with a clear error if `NEO4J_PASSWORD` isn't set — a
   mismatch with the graph's actual password (rather than a missing var) will
   instead surface as a Neo4j auth error at connection time.

5. **Generate entity embeddings** (required before the first run — `OncologyKGMM.py`
   loads `OncologyKG/kg_export/entity_embeddings.json` unconditionally and will
   error out if it doesn't exist yet):
   ```bash
   python generate_entity_embeddings.py
   ```
   Only needs rerunning when `OncologyKG/kg_export/nodes.json` changes (e.g.
   after a graph rebuild) — `run_alliance.bash` handles this automatically by
   comparing file mtimes.

6. **Run it:**
   ```bash
   python OncologyKGMM.py
   ```
   Questions are defined in `TEST_QUESTIONS` at the bottom of the file. Results
   are written to `output.csv` (question + generated answer per row).

## Citation

If you build on the original MindMap approach, please cite:

```latex
@inproceedings{wen2023mindmap,
  title={MindMap: Knowledge Graph Prompting Sparks Graph of Thoughts in Large Language Models},
  author={Wen, Yilin and Wang, Zifeng and Sun, Jimeng},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics},
  year={2024}
}
```
