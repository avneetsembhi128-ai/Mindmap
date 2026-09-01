"""
kg_parsers.py — reads the raw ClinPGx/SIDER/ClinVar files and turns each one
into a list of graph nodes and edges (triples). There are 8 sources, each
with its own parse_* function below; kg.py's cmd_build() calls all of them
and loads the combined result into Neo4j.
"""

import gzip
import os
import re
from collections import defaultdict

import pandas as pd

# ADR severity-grade lookup table, used only by the SIDER parser
from kg_constants import CTCAE_MAP
# Shared helpers: drug/ADR name matching, triple building, and cleanup functions
from kg_helpers import (
    make_triple, is_target_drug, canonicalize_drug, get_drug_class,
    split_multi_value_field, canonicalize_adr, is_target_adr,
    EVIDENCE_CONFIDENCE, _level_rank, split_variant_list,
    _extract_baseline_text, _normalize_for_match, _clean_str, _num_or_none,
)


# ── Source 1 — ClinPGx Genes ─────────────────────────────────────

# Reads genes.tsv and builds one Gene node per gene symbol
def parse_genes(path):
    df = pd.read_csv(path, sep="\t", low_memory=False)
    nodes = []
    for _, row in df.iterrows():
        symbol = str(row.get("Symbol", "")).strip()
        if not symbol or symbol == "nan":
            continue
        nodes.append({
            "name":      symbol,
            "full_name": str(row.get("Name", "")).strip(),
            "has_cpic":  str(row.get("Has CPIC Dosing Guideline", "No")),
            "label":     "Gene"
        })
    print(f"  Genes: {len(nodes):,}")
    return nodes


# ── Source 2 — ClinPGx Drugs ──────────────────────────────────────

# Reads drugs.tsv and builds one Drug node per target drug, with its
# brand-name/generic-name synonyms attached
def parse_drugs(path):
    df = pd.read_csv(path, sep="\t", low_memory=False)
    nodes = []
    seen_names = set()
    for _, row in df.iterrows():
        raw_name = str(row.get("Name", "")).strip()
        if not raw_name or raw_name == "nan":
            continue
        canonical = canonicalize_drug(raw_name)
        if not canonical:
            continue
        if canonical in seen_names:
            continue
        seen_names.add(canonical)
        synonyms = []
        for col in ("Generic Names", "Trade Names"):
            synonyms += split_multi_value_field(str(row.get(col, "")))
        nodes.append({
            "name":       canonical,
            "drug_class": get_drug_class(canonical),
            "drug_type":  str(row.get("Type", "")).strip(),
            "cpic_level": str(row.get("Top CPIC Pairs Level", "")).strip(),
            "source_terms": raw_name,
            "synonyms":   ", ".join(synonyms),
            "label":      "Drug"
        })
    print(f"  Drugs: {len(nodes):,}")
    return nodes


# ── Source 3 — ClinPGx Variants (rsID + gene symbol reference) ───

# Reads variants.tsv, a reference table of rsIDs and which gene(s) they fall in
def parse_variants(path):
    df = pd.read_csv(path, sep="\t", low_memory=False)
    nodes = []
    for _, row in df.iterrows():
        name = str(row.get("Variant Name", "")).strip()
        if not name or name == "nan":
            continue
        rsid = name if name.startswith("rs") else ""
        nodes.append({
            "name":         name,
            "rsid":         rsid,
            "gene_symbols": str(row.get("Gene Symbols", "")).strip(),
            "location":     str(row.get("Location", "")).strip(),
            "label":        "Variant"
        })
    print(f"  Variants: {len(nodes):,}")
    return nodes


# ── Source 4 — ClinPGx Clinical Variants (variant -> drug links) ─

# Reads clinicalVariants.tsv and links each variant to its gene, the
# drug(s) it affects, and any ADR(s) it's linked to
def parse_clinical_variants(path):
    df = pd.read_csv(path, sep="\t", low_memory=False)
    nodes   = []
    triples = []

    for _, row in df.iterrows():
        variant_raw = str(row.get("variant",  "")).strip()
        gene     = str(row.get("gene",     "")).strip()
        ev_level = str(row.get("level of evidence", "")).strip()
        chemicals= str(row.get("chemicals","")).strip()
        phenotypes_raw = str(row.get("phenotypes","")).strip()

        if not variant_raw or variant_raw == "nan":
            continue

        # Skip rows that don't touch a target drug or ADR at all
        drug_relevant = (chemicals != "nan" and chemicals and
                        is_target_drug(chemicals))
        adr_relevant  = (phenotypes_raw != "nan" and phenotypes_raw and
                        is_target_adr(phenotypes_raw))

        if not drug_relevant and not adr_relevant:
            continue

        conf = EVIDENCE_CONFIDENCE.get(ev_level, "low")

        for variant in split_variant_list(variant_raw):
            nodes.append({
                "name":  variant,
                "rsid":  variant if variant.startswith("rs") else "",
                "gene":  gene if gene and gene != "nan" else "",
                "source_terms": variant_raw if variant_raw != variant else "",
                "label": "Variant"
            })

            # Gene -> HAS_CLINICAL_VARIANT -> Variant
            if gene and gene != "nan":
                triples.append(make_triple(
                    gene, "Gene", "HAS_CLINICAL_VARIANT",
                    variant, "Variant",
                    "PharmGKB_clinicalVariants", conf,
                    evidence_level=ev_level
                ))

            # Variant -> AFFECTS_RESPONSE_TO -> Drug
            if chemicals and chemicals != "nan":
                for chem in split_multi_value_field(chemicals):
                    canonical_drug = canonicalize_drug(chem)
                    if canonical_drug:
                        triples.append(make_triple(
                            variant, "Variant", "AFFECTS_RESPONSE_TO",
                            canonical_drug, "Drug",
                            "PharmGKB_clinicalVariants", conf,
                            evidence_level=ev_level,
                            source_term=chem
                        ))

            # Variant -> LINKED_TO_ADR -> ADR
            if phenotypes_raw and phenotypes_raw != "nan":
                for phen in split_multi_value_field(phenotypes_raw):
                    canonical_adr = canonicalize_adr(phen)
                    if canonical_adr:
                        nodes.append({
                            "name": canonical_adr,
                            "source_terms": phen,
                            "label": "ADR"
                        })
                        triples.append(make_triple(
                            variant, "Variant", "LINKED_TO_ADR",
                            canonical_adr, "ADR",
                            "PharmGKB_clinicalVariants", conf,
                            evidence_level=ev_level,
                            source_term=phen
                        ))

    print(f"  Clinical variants: {len(triples):,} edges (filtered to oncology ADRs)")
    return nodes, triples


# ── Source 5 — ClinPGx Summary Annotations (real 1A-4 evidence grade) ──

def parse_summary_annotations(annotations_path, evidence_path):
    """Reads ClinPGx's curated 1A-4 evidence grade for a variant-drug/ADR
    finding — a stronger signal than the flat "medium" confidence other
    edges default to. The two files join through a shared ID (evidence's
    "Evidence ID" column matches a Variant Annotation ID), not directly.

    Returns {variant_annotation_id: level_of_evidence}. If one annotation
    is graded by more than one Summary Annotation, keeps the strongest.

    Not every annotation has a Summary Annotation grade — pediatric oncology
    is a newer, less-curated part of ClinPGx, so coverage here is lower than
    ClinPGx's database-wide average. That's an honest source-data gap, not
    a bug in this join.
    """
    summaries = pd.read_csv(annotations_path, sep="\t", low_memory=False)
    evidence  = pd.read_csv(evidence_path, sep="\t", low_memory=False)

    level_by_summary_id = {}
    for _, row in summaries.iterrows():
        sid = _clean_str(row.get("Summary Annotation ID"))
        level = _clean_str(row.get("Level of Evidence"))
        if sid and level:
            level_by_summary_id[sid] = level

    level_by_annotation_id = {}
    for _, row in evidence.iterrows():
        annotation_id = _clean_str(row.get("Evidence ID"))
        sid = _clean_str(row.get("Summary Annotation ID"))
        if not annotation_id or not sid:
            continue
        level = level_by_summary_id.get(sid)
        if not level:
            continue
        existing = level_by_annotation_id.get(annotation_id)
        if existing is None or _level_rank(level) < _level_rank(existing):
            level_by_annotation_id[annotation_id] = level

    print(f"  Summary annotations: {len(level_by_summary_id):,} summaries, "
          f"{len(level_by_annotation_id):,} Variant Annotation IDs with a real evidence grade")
    return level_by_annotation_id


# ── Source 6 — ClinPGx Variant Annotations ────────────────────────

# Reads var_drug_ann.tsv and links each variant to the drug(s) its
# pharmacogenomic association is about
def parse_variant_drug_annotations(path, evidence_level_map=None):
    df = pd.read_csv(path, sep="\t", low_memory=False)
    nodes   = []
    triples = []
    # Remembers which (variant, relation, drug/ADR) each annotation ID
    # produced, so parse_study_parameters() below can attach Study nodes
    # to the right edge later
    annotation_map = defaultdict(list)
    # PMID of the original paper each annotation came from — ClinPGx just
    # compiles these studies, it doesn't run them, so this is what lets us
    # cite the real source paper instead of just "ClinPGx" as the source
    pmid_by_annotation_id = {}
    evidence_level_map = evidence_level_map or {}

    for _, row in df.iterrows():
        annotation_id = _clean_str(row.get("Variant Annotation ID"))
        variant_raw = str(row.get("Variant/Haplotypes", "")).strip()
        gene     = str(row.get("Gene",     "")).strip()
        drug     = str(row.get("Drug(s)",  "")).strip()
        sentence = str(row.get("Sentence", "")).strip()
        pheno_cat= str(row.get("Phenotype Category", "")).strip()
        pdpk     = str(row.get("PD/PK terms", "")).strip()
        direction= str(row.get("Direction of effect", "")).strip()
        pmid     = _clean_str(row.get("PMID"))
        citation_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
        if annotation_id and pmid:
            pmid_by_annotation_id[annotation_id] = pmid

        if not variant_raw or variant_raw == "nan":
            continue
        if not drug or drug == "nan":
            continue
        if not is_target_drug(drug):
            continue

        # Use the real evidence grade if one exists; otherwise leave it
        # unknown (None) instead of guessing "medium"
        evidence_level = evidence_level_map.get(annotation_id) if annotation_id else None
        confidence = EVIDENCE_CONFIDENCE.get(evidence_level) if evidence_level else None

        # Guess the mechanism type from the PD/PK terms and sentence wording
        mechanism = "unknown"
        if pdpk and pdpk != "nan":
            if "metabolism" in pdpk.lower() or "pk" in pdpk.lower():
                mechanism = "pharmacokinetic"
            elif "pd" in pdpk.lower() or "efficacy" in pdpk.lower():
                mechanism = "pharmacodynamic"
        if "immune" in sentence.lower() or "hypersensitivity" in sentence.lower():
            mechanism = "immune_mediated"

        baseline_text = _extract_baseline_text(sentence)

        for variant in split_variant_list(variant_raw):
            # Skip the baseline allele/genotype — it's the reference group
            # being compared against, not a finding of its own
            if baseline_text and _normalize_for_match(variant) in _normalize_for_match(baseline_text):
                continue

            nodes.append({
                "name":      variant,
                "hgvs":      variant if variant.startswith("c.") or
                                        variant.startswith("p.") or
                                        variant.startswith("g.") else "",
                "star_allele": variant if "*" in variant else "",
                "rsid":      variant if variant.startswith("rs") else "",
                "gene":      gene if gene and gene != "nan" else "",
                "source_terms": variant_raw if variant_raw != variant else "",
                "label":     "Variant"
            })

            # Variant -> PHARMACOGENOMIC_ASSOCIATION -> Drug
            for d in split_multi_value_field(drug):
                canonical_drug = canonicalize_drug(d)
                if canonical_drug:
                    triples.append(make_triple(
                        variant, "Variant", "PHARMACOGENOMIC_ASSOCIATION",
                        canonical_drug, "Drug",
                        "PharmGKB_variantAnnotations", confidence,
                        mechanism=mechanism,
                        phenotype_category=pheno_cat,
                        direction=direction,
                        description=sentence if sentence and sentence != "nan" else "",
                        source_term=d,
                        variant_annotation_id=annotation_id,
                        evidence_level=evidence_level,
                        pmid=pmid,
                        citation_url=citation_url
                    ))
                    if annotation_id:
                        annotation_map[annotation_id].append(
                            (variant, "PHARMACOGENOMIC_ASSOCIATION",
                             canonical_drug, "Drug"))

            # Gene -> HAS_VARIANT -> Variant
            if gene and gene != "nan":
                for g in split_multi_value_field(gene):
                    triples.append(make_triple(
                        g, "Gene", "HAS_VARIANT",
                        variant, "Variant",
                        "PharmGKB_variantAnnotations", "medium",
                        mechanism=mechanism
                    ))

    print(f"  Variant-drug annotations: {len(triples):,} edges (filtered to oncology drugs)")
    return nodes, triples, annotation_map, pmid_by_annotation_id


# Reads var_pheno_ann.tsv and links each variant to the ADR(s)/phenotype(s)
# it's associated with
def parse_variant_pheno_annotations(path, evidence_level_map=None):
    df = pd.read_csv(path, sep="\t", low_memory=False)
    nodes   = []
    triples = []
    annotation_map = defaultdict(list)
    pmid_by_annotation_id = {}
    evidence_level_map = evidence_level_map or {}

    for _, row in df.iterrows():
        annotation_id = _clean_str(row.get("Variant Annotation ID"))
        variant_raw = str(row.get("Variant/Haplotypes", "")).strip()
        gene     = str(row.get("Gene",      "")).strip()
        drug     = str(row.get("Drug(s)",   "")).strip()
        phenotype= str(row.get("Phenotype", "")).strip()
        side_eff = str(row.get("Side effect/efficacy/other", "")).strip()
        sentence = str(row.get("Sentence",  "")).strip()
        direction= str(row.get("Direction of effect", "")).strip()
        pmid     = _clean_str(row.get("PMID"))
        citation_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
        if annotation_id and pmid:
            pmid_by_annotation_id[annotation_id] = pmid

        if not variant_raw or variant_raw == "nan":
            continue

        if not is_target_drug(drug) and not is_target_adr(phenotype):
            continue

        # Same evidence-grade lookup as parse_variant_drug_annotations above
        evidence_level = evidence_level_map.get(annotation_id) if annotation_id else None
        confidence = EVIDENCE_CONFIDENCE.get(evidence_level) if evidence_level else None

        baseline_text = _extract_baseline_text(sentence)

        for variant in split_variant_list(variant_raw):
            # Skip the baseline allele/genotype (see the comment in
            # parse_variant_drug_annotations above)
            if baseline_text and _normalize_for_match(variant) in _normalize_for_match(baseline_text):
                continue

            # Always create a Variant node here too — some variants (mostly
            # HLA alleles and named metabolizer phenotypes) only ever show
            # up in this file, so without this they'd have no node at all
            # and their ADR edge below would have nothing to attach to
            nodes.append({
                "name":      variant,
                "hgvs":      variant if variant.startswith("c.") or
                                        variant.startswith("p.") or
                                        variant.startswith("g.") else "",
                "star_allele": variant if "*" in variant else "",
                "rsid":      variant if variant.startswith("rs") else "",
                "gene":      gene if gene and gene != "nan" else "",
                "source_terms": variant_raw if variant_raw != variant else "",
                "label":     "Variant"
            })

            # Gene -> HAS_VARIANT -> Variant
            if gene and gene != "nan":
                for g in split_multi_value_field(gene):
                    triples.append(make_triple(
                        g, "Gene", "HAS_VARIANT",
                        variant, "Variant",
                        "PharmGKB_variantAnnotations", "medium"
                    ))

            if phenotype and phenotype != "nan":
                for phen in split_multi_value_field(phenotype):
                    canonical_adr = canonicalize_adr(phen)
                    if canonical_adr:
                        nodes.append({
                            "name": canonical_adr,
                            "source_terms": phen,
                            "label": "ADR"
                        })
                        triples.append(make_triple(
                            variant, "Variant", "ASSOCIATED_WITH_ADR",
                            canonical_adr, "ADR",
                            "PharmGKB_variantAnnotations", confidence,
                            side_effect_type=side_eff,
                            direction=direction,
                            description=sentence if sentence and sentence != "nan" else "",
                            source_term=phen,
                            variant_annotation_id=annotation_id,
                            evidence_level=evidence_level,
                            pmid=pmid,
                            citation_url=citation_url
                        ))
                        if annotation_id:
                            annotation_map[annotation_id].append(
                                (variant, "ASSOCIATED_WITH_ADR",
                                 canonical_adr, "ADR"))

    print(f"  Variant-phenotype annotations: {len(triples):,} edges (filtered to oncology ADRs)")
    return nodes, triples, annotation_map, pmid_by_annotation_id


# Reads var_fa_ann.tsv — lab/in-vitro findings (enzyme activity, cell
# assays), as opposed to the clinical findings the other two variant-
# annotation files carry. Kept as its own edge type (HAS_FUNCTIONAL_EVIDENCE)
# so a lab result never gets mistaken for a real clinical association.
def parse_variant_fa_annotations(path):
    df = pd.read_csv(path, sep="\t", low_memory=False)
    nodes   = []
    triples = []
    annotation_map = defaultdict(list)
    pmid_by_annotation_id = {}

    for _, row in df.iterrows():
        annotation_id = _clean_str(row.get("Variant Annotation ID"))
        variant_raw = str(row.get("Variant/Haplotypes", "")).strip()
        gene     = str(row.get("Gene",     "")).strip()
        drug     = str(row.get("Drug(s)",  "")).strip()
        sentence = str(row.get("Sentence", "")).strip()
        direction= str(row.get("Direction of effect", "")).strip()
        functional_terms = _clean_str(row.get("Functional terms")) or ""
        assay_type = _clean_str(row.get("Assay type")) or ""
        cell_type  = _clean_str(row.get("Cell type")) or ""
        pmid     = _clean_str(row.get("PMID"))
        citation_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
        if annotation_id and pmid:
            pmid_by_annotation_id[annotation_id] = pmid

        if not variant_raw or variant_raw == "nan":
            continue
        if not drug or drug == "nan":
            continue
        if not is_target_drug(drug):
            continue

        baseline_text = _extract_baseline_text(sentence)

        for variant in split_variant_list(variant_raw):
            if baseline_text and _normalize_for_match(variant) in _normalize_for_match(baseline_text):
                continue

            nodes.append({
                "name":      variant,
                "hgvs":      variant if variant.startswith("c.") or
                                        variant.startswith("p.") or
                                        variant.startswith("g.") else "",
                "star_allele": variant if "*" in variant else "",
                "rsid":      variant if variant.startswith("rs") else "",
                "gene":      gene if gene and gene != "nan" else "",
                "source_terms": variant_raw if variant_raw != variant else "",
                "label":     "Variant"
            })

            # Variant -> HAS_FUNCTIONAL_EVIDENCE -> Drug
            for d in split_multi_value_field(drug):
                canonical_drug = canonicalize_drug(d)
                if canonical_drug:
                    triples.append(make_triple(
                        variant, "Variant", "HAS_FUNCTIONAL_EVIDENCE",
                        canonical_drug, "Drug",
                        "PharmGKB_variantAnnotations_functional", "medium",
                        direction=direction,
                        functional_terms=functional_terms,
                        assay_type=assay_type,
                        cell_type=cell_type,
                        description=sentence if sentence and sentence != "nan" else "",
                        source_term=d,
                        variant_annotation_id=annotation_id,
                        pmid=pmid,
                        citation_url=citation_url
                    ))
                    if annotation_id:
                        annotation_map[annotation_id].append(
                            (variant, "HAS_FUNCTIONAL_EVIDENCE",
                             canonical_drug, "Drug"))

            # Gene -> HAS_VARIANT -> Variant
            if gene and gene != "nan":
                for g in split_multi_value_field(gene):
                    triples.append(make_triple(
                        g, "Gene", "HAS_VARIANT",
                        variant, "Variant",
                        "PharmGKB_variantAnnotations_functional", "medium"
                    ))

    print(f"  Variant functional-assay annotations: {len(triples):,} edges (filtered to oncology drugs)")
    return nodes, triples, annotation_map, pmid_by_annotation_id


# Reads ClinPGx's pediatric dashboard export — every row in this file is
# an annotation ID that's been curator-confirmed as pediatric-relevant.
# An ID NOT in this set doesn't mean "adult" — it could just mean nobody
# has reviewed it for pediatric relevance yet, so absence is treated as
# "unknown", not "no", wherever this set gets used.
def parse_pediatric_tags(path):
    df = pd.read_csv(path, sep="\t", low_memory=False)
    tagged_ids = {_clean_str(v) for v in df.get("ID", []) if _clean_str(v)}
    print(f"  Pediatric tags: {len(tagged_ids):,} pediatric-tagged Variant Annotation IDs")
    return tagged_ids


# Reads study_parameters.tsv — the actual study design behind a finding
# (effect size, confidence interval, sample size, study type). This file
# has no variant/drug/ADR columns of its own, so every row is matched up
# to a finding purely through its Variant Annotation ID, using the
# annotation_map built while parsing the 3 variant-annotation files above.
# pmid_map (also built from those 3 files) attaches the real published
# paper's PMID/citation link to each Study node.
def parse_study_parameters(path, annotation_map, pediatric_ids=None, pmid_map=None):
    df = pd.read_csv(path, sep="\t", low_memory=False)
    pediatric_ids = pediatric_ids or set()
    pmid_map = pmid_map or {}
    nodes   = []
    triples = []
    linked_edges = 0
    unmatched = 0

    # Best-effort only — PharmGKB doesn't guarantee either field is
    # structured in Characteristics free text. Left as None/absent (not
    # False) when not confidently found: absence here is an honest data
    # gap, not evidence the study lacked that detail.
    age_pattern = re.compile(
        r"\b(\d{1,3})\s*(?:-|–|to)\s*(\d{1,3})\s*(?:years|yrs|y\.?o\.?)\b",
        re.IGNORECASE)
    dosing_pattern = re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:mg/kg|mg/m\^?2|mg/m2|mg|mcg|g)\b",
        re.IGNORECASE)

    for _, row in df.iterrows():
        study_id = _clean_str(row.get("Study Parameters ID"))
        annotation_id = _clean_str(row.get("Variant Annotation ID"))
        if not study_id or not annotation_id:
            continue

        findings = annotation_map.get(annotation_id)
        if not findings:
            unmatched += 1
            continue

        characteristics = _clean_str(row.get("Characteristics"))
        age_match = age_pattern.search(characteristics) if characteristics else None
        age_range = f"{age_match.group(1)}-{age_match.group(2)} years" if age_match else None
        dosing_reported = True if (characteristics and dosing_pattern.search(characteristics)) else None
        # Task 5: real curator-assessed flag, kept alongside (not replacing)
        # the regex-guessed age_range above — the two aren't fully
        # redundant (age_range gives an actual bracket when parseable;
        # this gives a reliable yes/unknown regardless of whether the free
        # text happened to contain a parseable range at all).
        # Kept alongside the regex-guessed age_range above, not instead of
        # it — this gives a reliable yes/unknown even when the free text
        # has no parseable age range in it
        pediatric_tagged = True if annotation_id in pediatric_ids else None
        pmid = pmid_map.get(annotation_id)
        citation_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None

        study_name = f"PharmGKB Study {study_id}"
        nodes.append({
            "name":             study_name,
            "study_type":       _clean_str(row.get("Study Type")),
            "n_cases":          _num_or_none(row.get("Study Cases")),
            "n_controls":       _num_or_none(row.get("Study Controls")),
            "characteristics":  characteristics,
            "effect_size":      _num_or_none(row.get("Ratio Stat")),
            "effect_size_type": _clean_str(row.get("Ratio Stat Type")),
            "p_value":          _clean_str(row.get("P Value")),
            "ci_lower":         _num_or_none(row.get("Confidence Interval Start")),
            "ci_upper":         _num_or_none(row.get("Confidence Interval Stop")),
            "population":       _clean_str(row.get("Biogeographical Groups")),
            "pmid":             pmid,
            "citation_url":     citation_url,
            "age_range":        age_range,
            "pediatric_tagged": pediatric_tagged,
            "dosing_reported":  dosing_reported,
            "variant_annotation_id": annotation_id,
            "label":            "Study"
        })

        # Variant -> SUPPORTED_BY_STUDY -> Study, once per finding this
        # study supports. for_tail records WHICH finding (e.g. which ADR),
        # since one variant can have several evidence edges and one study
        # can back more than one of them at once.
        # Important: don't drop for_tail — without it, Neo4j would collapse
        # all of a variant's studies into a single edge and silently lose
        # every finding but the last one written (this really happened once).
        for (variant_name, relation, tail_name, tail_label) in findings:
            triples.append(make_triple(
                variant_name, "Variant", "SUPPORTED_BY_STUDY",
                study_name, "Study",
                "PharmGKB_studyParameters", "medium",
                for_relation=relation,
                for_tail=tail_name,
                for_tail_label=tail_label
            ))
        linked_edges += len(findings)

    print(f"  Study parameters: {len(nodes):,} Study nodes, {linked_edges:,} "
          f"SUPPORTED_BY_STUDY edges ({unmatched:,} rows had no matching "
          f"in-scope annotation)")
    return nodes, triples


# ── Source 7 — SIDER ──────────────────────────────────────────────

# Reads SIDER's side-effect data and links each target drug to the ADRs
# it's reported to cause
def parse_sider(se_path, names_path):
    id_to_name = {}
    if os.path.exists(names_path):
        names_df = pd.read_csv(names_path, sep="\t", header=None,
                               names=["stitch_id", "drug_name"],
                               low_memory=False)
        id_to_name = dict(zip(
            names_df["stitch_id"].astype(str),
            names_df["drug_name"].astype(str)
        ))

    target_stitch_ids = set()
    stitch_to_canonical = {}
    for stitch_id, drug_name in id_to_name.items():
        canonical = canonicalize_drug(drug_name)
        if canonical:
            target_stitch_ids.add(stitch_id)
            stitch_to_canonical[stitch_id] = canonical

    print(f"  SIDER target drug STITCH IDs found: {len(target_stitch_ids)}")

    nodes, triples = [], []
    seen = set()

    with gzip.open(se_path, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            stitch_flat = parts[0]
            side_effect = parts[5] if len(parts) > 5 else ""

            if stitch_flat not in target_stitch_ids:
                continue

            canonical_adr = canonicalize_adr(side_effect)
            if not canonical_adr:
                continue

            canonical_drug = stitch_to_canonical.get(stitch_flat)
            if not canonical_drug:
                continue

            key = (canonical_drug, canonical_adr)
            if key in seen:
                continue
            seen.add(key)

            ctcae_info = CTCAE_MAP.get(canonical_adr, {})

            nodes.append({
                "name":       canonical_adr,
                "source_terms": side_effect,
                "ctcae_term": ctcae_info.get("term", ""),
                "ctcae_grades": ctcae_info.get("grades", ""),
                "label":      "ADR"
            })

            triples.append(make_triple(
                canonical_drug, "Drug", "HAS_SIDE_EFFECT",
                canonical_adr, "ADR",
                "SIDER", "medium",
                source_term=side_effect,
                ctcae_term=ctcae_info.get("term", ""),
                ctcae_grades=ctcae_info.get("grades", "")
            ))

    print(f"  SIDER: {len(triples):,} drug-ADR pairs for oncology drugs")
    return nodes, triples


# ── Source 8 — ClinVar ────────────────────────────────────────────

# Reads ClinVar's variant summary file and links pharmacogenes to their
# variants and clinical severity — a large gzipped file, so this can take
# a few minutes to run
def parse_clinvar(path):
    # Only these genes are relevant to this project's target drugs/ADRs
    TARGET_GENES = {
        "TPMT", "COMT", "ACYP2", "ABCC3", "LRP2",
        "CBR1", "CBR3", "RARG", "SLC28A3", "UGT1A6",
        "CEP72", "ABCB1", "CYP3A5",
        "MTHFR", "ABCC2", "SLC19A1", "TYMS",
        "CYP2C8", "EPHA4", "EPHA5",
    }

    SEVERITY_MAP = {
        "pathogenic":                    "high",
        "likely pathogenic":             "medium-high",
        "pathogenic/likely pathogenic":  "high",
        "risk factor":                   "medium",
        "drug response":                 "medium",
        "uncertain significance":        "low",
    }

    print("  Parsing ClinVar (large file, may take a few minutes)...")
    nodes   = []
    triples = []
    seen    = set()

    with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as f:
        header = None
        for i, line in enumerate(f):
            if i == 0:
                header = line.strip().split("\t")
                continue
            parts = line.strip().split("\t")
            if len(parts) < len(header):
                continue

            row = dict(zip(header, parts))
            gene     = str(row.get("GeneSymbol", "")).strip()
            rsid     = str(row.get("RS# (dbSNP)", "")).strip()
            clin_sig = str(row.get("ClinicalSignificance", "")).lower().strip()
            phenotype_list_raw = str(row.get("PhenotypeList", "")).strip()
            name     = str(row.get("Name", "")).strip()

            if gene not in TARGET_GENES:
                continue
            if not rsid or rsid == "-1" or rsid == "nan":
                continue

            canonical_adrs = []
            if phenotype_list_raw and phenotype_list_raw != "nan":
                for phen in phenotype_list_raw.split("|"):
                    phen = phen.strip()
                    if not phen or phen.lower() in ("not specified", "not provided"):
                        continue
                    canonical = canonicalize_adr(phen)
                    if canonical:
                        canonical_adrs.append((canonical, phen))

            if not canonical_adrs and "drug" not in clin_sig:
                continue

            rsid_fmt = f"rs{rsid}" if not rsid.startswith("rs") else rsid
            key = (rsid_fmt, gene)
            if key in seen:
                continue
            seen.add(key)

            severity = SEVERITY_MAP.get(clin_sig, "low")

            nodes.append({
                "name":               rsid_fmt,
                "rsid":               rsid_fmt,
                "hgvs":               name,
                "clinical_significance": clin_sig,
                "severity":           severity,
                "label":              "Variant"
            })

            triples.append(make_triple(
                rsid_fmt, "Variant", "IN_GENE",
                gene, "Gene",
                "ClinVar", severity,
                clinical_significance=clin_sig
            ))

            for canonical_adr, source_term in canonical_adrs:
                nodes.append({
                    "name": canonical_adr,
                    "source_terms": source_term,
                    "label": "ADR"
                })
                triples.append(make_triple(
                    rsid_fmt, "Variant", "CLINVAR_ASSOCIATED_ADR",
                    canonical_adr, "ADR",
                    "ClinVar", severity,
                    clinical_significance=clin_sig,
                    source_term=source_term
                ))

    print(f"  ClinVar: {len(nodes):,} variants, {len(triples):,} edges for oncology genes")
    return nodes, triples
