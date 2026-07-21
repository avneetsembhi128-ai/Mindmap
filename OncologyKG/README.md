# OncologyKG

A focused Pediatric Oncology Adverse Drug Reaction (ADR) knowledge graph in Neo4j,
built from PharmGKB, CPIC, SIDER, and ClinVar. It links Genes → Variants → Drugs → ADRs
for a curated set of chemotherapy agents and clinically significant toxicities
(ototoxicity, cardiotoxicity, peripheral neuropathy, mucositis, hepatotoxicity,
neutropenia, thrombocytopenia, myelosuppression, nephrotoxicity, hypersensitivity).

Everything — build, load, export, audit — lives in one CLI, [`kg.py`](kg.py).

## Two ways to reproduce the graph

### Option A — Reload the pre-built graph (fast, no source data needed)

The graph is exported to `kg_export/nodes.json` and `kg_export/edges.json`, both
committed to this repo. This is the easiest way to get the exact graph used for
downstream work, with no PharmGKB/CPIC/SIDER/ClinVar downloads required.

```bash
pip install -r requirements.txt

# Start a local Neo4j instance (Desktop or Docker), then:
export NEO4J_PASSWORD="your-password-here"   # PowerShell: $env:NEO4J_PASSWORD = "..."
python kg.py load
```

### Option B — Rebuild from raw source data

Use this if you want to verify the build logic, pull fresher source data, or change
the target drug/ADR scope (edit `TARGET_DRUGS` / `ADR_CANONICAL_MAP` near the top of
`kg.py`).

1. Install dependencies: `pip install -r requirements.txt`
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

None of these files are committed (`data/` is gitignored). Download each into the
path shown, relative to `OncologyKG/data/`.

| File(s) | Source | Path |
|---|---|---|
| `genes.tsv`, `drugs.tsv`, `variants.tsv`, `clinicalVariants.tsv` | [PharmGKB downloads](https://www.pharmgkb.org/downloads) | `data/genes/genes.tsv`, `data/drugs/drugs.tsv`, `data/variants/variants.tsv`, `data/clinicalVariants/clinicalVariants.tsv` |
| `var_drug_ann.tsv`, `var_pheno_ann.tsv`, `var_fa_ann.tsv`, `study_parameters.tsv` | PharmGKB downloads → "Variant Annotations" zip | `data/variantAnnotations/` |
| `cpic_recommendations.json`, `cpic_drugs.json` | [CPIC public API](https://api.cpicpgx.org/v1/) — `recommendation` and `drug` endpoints | `data/cpic_recommendations.json`, `data/cpic_drugs.json` |
| `SIDER_side_effects.tsv.gz`, `SIDER_drug_names.tsv` | [SIDER 4.1](http://sideeffects.embl.de/) — `meddra_all_se.tsv.gz` (rename) and `drug_names.tsv` | `data/SIDER_side_effects.tsv.gz`, `data/SIDER_drug_names.tsv` |
| `clinvar_variant_summary.txt.gz` | [NCBI ClinVar FTP](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz) | `data/clinvar_variant_summary.txt.gz` |

**Note on reproducibility:** these are all live, continuously-updated databases, so a
rebuild today will not byte-for-byte match a rebuild from months ago — record the
download date when you fetch a fresh snapshot. The data currently used for this repo's
`kg_export/` was downloaded in June–July 2025. For a fully pinned reproduction, prefer
**Option A**.

## Folder layout

```
OncologyKG/
  kg.py              build / load / export / audit — one CLI, one shared
                      Neo4j connection helper
  requirements.txt
  README.md
  data/              raw source data (gitignored, see Data sources above)
  kg_export/
    nodes.json       committed — the portable graph snapshot
    edges.json
```

## Requirements

- Python 3.9+
- A running Neo4j instance (Desktop, Docker, or Aura) reachable at `neo4j://127.0.0.1:7687`
  (override with the `NEO4J_URI` / `NEO4J_USER` environment variables if yours differs)
- `NEO4J_PASSWORD` environment variable set before running any `kg.py` subcommand
