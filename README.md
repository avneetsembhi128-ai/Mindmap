# MindMap for Pediatric Oncology ADR

This is an adaptation of **MindMap** — [Knowledge Graph Prompting Sparks Graph of
Thoughts in Large Language Models](https://arxiv.org/pdf/2308.09729.pdf) (Wen et al.,
ACL'24) — retargeted to synthesize published research on pediatric oncology adverse
drug reactions (ADRs), using a purpose-built knowledge graph instead of the paper's
original chatdoctor5k dataset.

## How it fits together

```
OncologyKG/            builds and hosts the knowledge graph (Neo4j)
  kg.py                  build / load / export / audit — see OncologyKG/README.md
  kg_export/              committed graph snapshot (nodes.json, edges.json)

OncologyKGMM.py         MindMap's graph-of-thoughts pipeline, adapted to query
                         OncologyKG instead of chatdoctor5k
```

`OncologyKGMM.py` extracts entities from a clinical question with a local LLM,
looks them up in the Gene/Drug/Variant/ADR/Phenotype graph, finds paths and
neighbor evidence between matched entities, and asks the LLM to synthesize a
grounded answer — the same reasoning flow as the original MindMap, just pointed
at this graph.

## Run it

1. **Build/load the graph.** See [OncologyKG/README.md](OncologyKG/README.md) —
   fastest path is `cd OncologyKG && python kg.py load`, which needs a running
   Neo4j instance and `NEO4J_PASSWORD` set. Remember the password you use here —
   step 4 below must match it.

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Have an LLM endpoint available.** By default `OncologyKGMM.py` talks to a
   local Ollama server serving `qwen3:8b`. Pull the model first
   (`ollama pull qwen3:8b`), or point at a different OpenAI-compatible endpoint
   via env vars (see below) instead of editing the script.

4. **Set environment variables** (same convention as `OncologyKG/kg.py` — nothing
   is hardcoded, so this reproduces on another machine without touching source):

   | Variable | Default | Purpose |
   |---|---|---|
   | `NEO4J_PASSWORD` | *(required, no default)* | Must match whatever password you used for `kg.py load`/`build` in step 1 |
   | `NEO4J_URI` | `neo4j://127.0.0.1:7687` | Neo4j connection URI |
   | `NEO4J_USER` | `neo4j` | Neo4j username |
   | `LLM_BASE_URL` | `http://127.0.0.1:11434/v1` | OpenAI-compatible chat completions endpoint |
   | `LLM_MODEL` | `qwen3:8b` | Model name passed to that endpoint |

   ```bash
   # PowerShell
   $env:NEO4J_PASSWORD = "your-password-here"
   # bash
   export NEO4J_PASSWORD="your-password-here"
   ```

   The script exits with a clear error if `NEO4J_PASSWORD` isn't set — a
   mismatch with the graph's actual password (rather than a missing var) will
   instead surface as a Neo4j auth error at connection time.

5. **Run it:**
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
