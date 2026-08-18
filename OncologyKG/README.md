# OncologyKG

A focused Pediatric Oncology Adverse Drug Reaction (ADR) knowledge graph in Neo4j,
built from 3 independent sources: ClinPGx, SIDER, and ClinVar. ClinPGx is the
2024-2025 merger of PharmGKB, CPIC, and PharmCAT under one platform — those are
not three separate sources, so `data/` and this doc treat them as the single
resource they now are. The graph links Genes → Variants → Drugs → ADRs, and each
evidence edge can carry `Study` nodes (effect size, CI, sample size, study design)
so a finding's strength can be judged, not just its direction, for a curated set
of chemotherapy agents and clinically significant toxicities (ototoxicity,
cardiotoxicity, peripheral neuropathy, mucositis, hepatotoxicity, neutropenia,
thrombocytopenia, myelosuppression, nephrotoxicity, hypersensitivity).

Everything — build, load, export, audit — lives in one CLI, [`kg.py`](kg.py).

## Two ways to reproduce the graph

### Option A — Reload the pre-built graph (fast, no source data needed)

The graph is exported to `kg_export/nodes.json` and `kg_export/edges.json`, both
committed to this repo. This is the easiest way to get the exact graph used for
downstream work, with no ClinPGx/SIDER/ClinVar downloads required.

```bash
pip install -r ../requirements.txt

# Start a local Neo4j instance (Desktop or Docker), then:
export NEO4J_PASSWORD="your-password-here"   # PowerShell: $env:NEO4J_PASSWORD = "..."
python kg.py load
```

### Option B — Rebuild from raw source data

Use this if you want to verify the build logic, pull fresher source data, or change
the target drug/ADR scope (edit `TARGET_DRUGS` / `ADR_CANONICAL_MAP` near the top of
`kg.py`).

1. Install dependencies: `pip install -r ../requirements.txt`
2. Download the raw source files (see **Data sources** below) into `data/` matching
   the layout listed there. `data/` is gitignored — it is not committed because
   ClinVar alone is ~420 MB.
3. Start Neo4j and set the password:
   ```bash
   export NEO4J_PASSWORD="your-password-here"
   ```
4. Run the build (this **clears and repopulates** the target database):
   ```bash
   python kg.py build
   ```
5. (Optional) Run the health-check audit:
   ```bash
   python kg.py audit
   ```
6. (Optional) Re-export the graph so `kg_export/` reflects the new build:
   ```bash
   python kg.py export
   ```

## Data sources

None of these files are committed (`data/` is gitignored). `data/` is organized
by **source**, not file type — see `data/README.md` for why. Download each file
into the path shown, relative to `OncologyKG/data/`.

| File(s) | Source | Path |
|---|---|---|
| `genes.tsv`, `drugs.tsv`, `variants.tsv`, `clinicalVariants.tsv` | [PharmGKB downloads](https://www.pharmgkb.org/downloads) (now ClinPGx) | `data/clinpgx/genes/genes.tsv`, `data/clinpgx/drugs/drugs.tsv`, `data/clinpgx/variants/variants.tsv`, `data/clinpgx/clinicalVariants/clinicalVariants.tsv` |
| `var_drug_ann.tsv`, `var_pheno_ann.tsv`, `var_fa_ann.tsv`, `study_parameters.tsv` | PharmGKB/ClinPGx downloads → "Variant Annotations" zip | `data/clinpgx/variantAnnotations/` |
| `summary_annotations.tsv`, `summary_ann_evidence.tsv` | PharmGKB/ClinPGx downloads → "Clinical Annotations" zip — real 1A-4 evidence grade per finding | `data/clinpgx/summaryAnnotations/` |
| `pediatric_variant_annotations.tsv` | ClinPGx's pediatric dashboard export — curator-assessed pediatric-population flag | `data/clinpgx/pediatric/pediatric_variant_annotations.tsv` |
| Pathway TSVs (Platinum, Doxorubicin, Methotrexate, Vinka Alkaloid, Taxane) | ClinPGx pathway diagrams → "Pathways" bulk download, TSV variant — drug-specific PK/PD grounding for mechanism narratives | `data/clinpgx/pathways/pathways-tsv/` |
| `pharmgkb-gene-drug-pairs.tsv` | [PharmGKB downloads](https://www.pharmgkb.org/downloads) — curated CPIC/DPWG/FDA gene-drug actionability table, used only by `OncologyKGMM.py` (not ingested into Neo4j) | `data/clinpgx/pairs/pharmgkb-gene-drug-pairs.tsv` |
| `SIDER_side_effects.tsv.gz`, `SIDER_drug_names.tsv` | [SIDER 4.1](http://sideeffects.embl.de/) — `meddra_all_se.tsv.gz` (rename) and `drug_names.tsv` | `data/sider/SIDER_side_effects.tsv.gz`, `data/sider/SIDER_drug_names.tsv` |
| `clinvar_variant_summary.txt.gz` | [NCBI ClinVar FTP](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz) | `data/clinvar/clinvar_variant_summary.txt.gz` |

**Note on reproducibility:** these are all live, continuously-updated databases, so a
rebuild today will not byte-for-byte match a rebuild from months ago — record the
download date when you fetch a fresh snapshot. The data currently used for this repo's
`kg_export/` was downloaded in June–July 2025. For a fully pinned reproduction, prefer
**Option A**.

## (Optional) Pre-generate mechanism narratives

`OncologyKGMM.py` explains WHY a variant contributes to an ADR — e.g. "GSTT1
null → less enzyme → cisplatin metabolites persist → cochlear damage" — not
just that the two are associated. It can synthesize these on the spot for
whatever entities a given question matches, but pre-generating them with
`enrich_mechanisms.py` avoids repeat LLM calls at query time and lets you
skim a QA sample before trusting the output. Five separate resumable,
cacheable steps (see the module docstring for the full rationale, including
why steps are split across login/compute nodes on a cluster without internet
on compute nodes):

```bash
python enrich_mechanisms.py fetch-gene-functions   # login node (needs internet — queries UniProt + NCBI)
python enrich_mechanisms.py draft-adr-pathways     # compute node (needs Ollama)
python enrich_mechanisms.py synthesize             # compute node (needs Ollama)
python enrich_mechanisms.py apply                  # either (no network/LLM) — writes kg_export/edges.json
python enrich_mechanisms.py qa-sample              # either (no network/LLM) — writes a review sample
```

After `apply`, rerun `python kg.py load` to push the enriched `edges.json`
into Neo4j. Every step caches its results under `kg_export/` (see Folder
layout below) and is safe to interrupt and rerun — already-cached entries
are skipped.

## Folder layout

```
OncologyKG/
  kg.py                          build / load / export / audit — one CLI, one
                                  shared Neo4j connection helper
  enrich_mechanisms.py           (optional) pre-generates mechanism narratives —
                                  see above
  README.md
  data/                          raw source data (gitignored, see Data sources above)
    README.md                      why data/ is organized by source, not file type
    clinpgx/                       PharmGKB/CPIC/PharmCAT — one merged platform
      pediatric/                     pediatric dashboard export (curator-assessed flag)
      pathways/pathways-tsv/         per-drug PK/PD pathway diagrams
      pairs/                         curated CPIC/DPWG/FDA gene-drug actionability table
    sider/
    clinvar/
  kg_export/
    nodes.json                     committed — the portable graph snapshot
    edges.json                     committed — gains a mechanism_narrative
                                    property per edge once enrich_mechanisms.py
                                    apply has been run
    entity_embeddings.json         generated by ../generate_entity_embeddings.py
    gene_function_cache.json       enrich_mechanisms.py cache — UniProt gene function text
    gene_biotype_cache.json        enrich_mechanisms.py cache — NCBI gene biotype
                                    (protein-coding / rRNA / tRNA / pseudo / ...),
                                    gates which genes UniProt is actually a valid
                                    function source for
    adr_pathway_cache.json         enrich_mechanisms.py cache — general injury
                                    pathway per ADR
    mechanism_narrative_cache.json enrich_mechanisms.py cache — the actual
                                    per-variant/ADR narratives
    mechanism_narrative_qa_sample.md  written by `qa-sample`, for manual review
```

Dependencies live in the top-level [`requirements.txt`](../requirements.txt) — it's a
superset covering both `kg.py` and the MindMap pipeline (`OncologyKGMM.py`) that
consumes this graph, so one `pip install` covers the whole project.

## Requirements

- Python 3.9+
- A running Neo4j instance (Desktop, Docker, or Aura) reachable at `neo4j://127.0.0.1:7687`
  (override with the `NEO4J_URI` / `NEO4J_USER` environment variables if yours differs)
- `NEO4J_PASSWORD` environment variable set before running any `kg.py` subcommand
