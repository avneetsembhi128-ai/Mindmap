# data/ layout

This folder is organized by **source**, not by file type:

```
data/
  clinpgx/    genes/, drugs/, variants/, clinicalVariants/, variantAnnotations/
  sider/      SIDER_side_effects.tsv.gz, SIDER_drug_names.tsv
  clinvar/    clinvar_variant_summary.txt.gz
```

## Why "clinpgx" and not "pharmgkb"

`clinpgx/` holds every file downloaded from what used to be pharmgkb.org.
ClinPGx is the 2024-2025 merger of **PharmGKB, CPIC, and PharmCAT** into one
platform — they are not three independent evidence sources any more, even
though `genes.tsv`, `variants.tsv`, `var_drug_ann.tsv`, `study_parameters.tsv`,
etc. all still look like "PharmGKB files" by name. Grouping them under one
`clinpgx/` folder (rather than, say, a separate `cpic/` folder alongside
`pharmgkb/`) keeps that fact visible in the directory tree itself, instead of
letting the shape of `data/` quietly imply more independent sources than the
graph actually has. See the top-level `OncologyKG/README.md` for the full
source table and download links.

`data/` itself is gitignored (raw source data is large and regenerable via
`kg.py build`) — this file is the one exception, kept committed so anyone
cloning fresh still sees the reasoning behind the folder names.
