"""
kg.py — unified CLI for OncologyKG

Builds and manages the Pediatric Oncology ADR Knowledge Graph (Gene -> Variant
-> Drug -> ADR) in Neo4j, sourced from PharmGKB, CPIC, SIDER, and ClinVar.

Subcommands:
    python kg.py load      Rebuild the graph from kg_export/ (committed to the
                            repo — no raw source data needed). This is the
                            fast path to reproduce the exact graph on a new
                            machine.
    python kg.py build     Rebuild the graph from scratch by parsing raw
                            PharmGKB/CPIC/SIDER/ClinVar files in data/ (see
                            README.md for where to download them — data/ is
                            gitignored, not committed).
    python kg.py export    Dump the live graph in Neo4j to kg_export/
                            nodes.json + edges.json, so it can be committed
                            and reloaded elsewhere via `load`.
    python kg.py audit     Health-check an existing graph: node counts,
                            fragmentation/orphan/duplicate scans, ADR and
                            drug connectivity, and the 6 primary drug->ADR
                            reasoning-chain checks.

All subcommands require NEO4J_PASSWORD to be set in the environment:
    PowerShell:  $env:NEO4J_PASSWORD = "your-password-here"
    bash:        export NEO4J_PASSWORD="your-password-here"
"""

import argparse
import gzip
import json
import os
import re
from collections import defaultdict

import pandas as pd
from neo4j import GraphDatabase

# ─────────────────────────────────────────────────────────────
# CONNECTION / PATHS — shared by every subcommand
# ─────────────────────────────────────────────────────────────
NEO4J_URI  = os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")
EXPORT_DIR = os.path.join(SCRIPT_DIR, "kg_export")

BATCH_SIZE = 500
LABELS = ["Gene", "Drug", "Variant", "ADR", "Phenotype"]


def get_driver():
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise SystemExit(
            "Set the NEO4J_PASSWORD environment variable before running this script.\n"
            "PowerShell:  $env:NEO4J_PASSWORD = \"your-password-here\"\n"
            "bash:        export NEO4J_PASSWORD=\"your-password-here\""
        )
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, password))


# ═════════════════════════════════════════════════════════════
# BUILD — rebuild the graph from raw PharmGKB/CPIC/SIDER/ClinVar source data
# ═════════════════════════════════════════════════════════════
#
# Focused on these 6 drug-ADR pairs:
#   cisplatin       -> ototoxicity
#   doxorubicin     -> cardiotoxicity  (anthracycline)
#   vincristine     -> peripheral neuropathy
#   methotrexate    -> mucositis
#   methotrexate    -> hepatotoxicity
#   paclitaxel      -> peripheral neuropathy
#
# ...expanded to standard pediatric-oncology (COG-protocol) chemotherapy
# agents and a broader set of clinically significant ADR categories. Both
# dicts below are the actual scope boundary: add/remove a drug or ADR keyword
# group here and every parsing function below picks it up automatically,
# since they all route through canonicalize_drug() / canonicalize_adr()
# rather than checking a hardcoded list per-function.

TARGET_DRUGS = {
    # Original 6-pair drugs
    "cisplatin":     "Platinum compound",
    "doxorubicin":   "Anthracycline",
    "vincristine":   "Vinca alkaloid",
    "methotrexate":  "Antimetabolite",
    "paclitaxel":    "Taxane",
    # Broadened to other standard pediatric-oncology (COG-protocol)
    # chemotherapy agents. Hand-curated, not derived from a PharmGKB
    # indication/ATC field — easy to prune or extend as a plain dict.
    "carboplatin":   "Platinum compound",
    "daunorubicin":  "Anthracycline",
    "epirubicin":    "Anthracycline",
    "idarubicin":    "Anthracycline",
    "vinblastine":   "Vinca alkaloid",
    "cyclophosphamide": "Alkylating agent",
    "ifosfamide":    "Alkylating agent",
    "busulfan":      "Alkylating agent",
    "melphalan":     "Alkylating agent",
    "etoposide":     "Topoisomerase inhibitor",
    "irinotecan":    "Topoisomerase inhibitor",
    "topotecan":     "Topoisomerase inhibitor",
    "cytarabine":    "Antimetabolite",
    "mercaptopurine":"Antimetabolite",
    "thioguanine":   "Antimetabolite",
    "fludarabine":   "Antimetabolite",
    "bleomycin":     "Antitumor antibiotic",
    "dactinomycin":  "Antitumor antibiotic",
    "asparaginase":  "Enzyme (asparagine-depleting)",
    "temozolomide":  "Alkylating agent",
    "dexrazoxane":   "Cardioprotectant",
    "imatinib":      "Tyrosine kinase inhibitor",
    "rituximab":     "Monoclonal antibody",
}

# Aliases map to the CANONICAL drug name they should merge into. E.g.
# "adriamycin" is a brand/alt name for doxorubicin — it should never become
# its own node.
DRUG_ALIASES = {
    "adriamycin":    "doxorubicin",
    "vp-16":         "etoposide",
    "vp16":          "etoposide",
    "ara-c":         "cytarabine",
    "cytosar":       "cytarabine",
    "6-mp":          "mercaptopurine",
    "6-mercaptopurine": "mercaptopurine",
    "6-tg":          "thioguanine",
    "6-thioguanine": "thioguanine",
    "ctx":           "cyclophosphamide",
    "cytoxan":       "cyclophosphamide",
    "cpt-11":        "irinotecan",
}

# Canonical ADR categories, keyed by the keywords used to both filter rows
# and decide the canonical NODE NAME. Order matters: first match wins, so
# more specific keywords are listed first within reason.
ADR_CANONICAL_MAP = {
    "ototox":               "Ototoxicity",
    "hearing":              "Ototoxicity",
    "tinnitus":              "Ototoxicity",
    "cardiotox":            "Cardiotoxicity",
    "cardiomyopath":        "Cardiotoxicity",
    "heart failure":        "Cardiotoxicity",
    "cardiac":              "Cardiotoxicity",
    "peripheral neuropath": "Peripheral Neuropathy",
    "neuropath":            "Peripheral Neuropathy",
    "mucositis":            "Mucositis",
    "stomatitis":           "Mucositis",
    "oral mucos":           "Mucositis",
    "hepatotox":            "Hepatotoxicity",
    "hepatic":              "Hepatotoxicity",
    "liver":                "Hepatotoxicity",
    "febrile neutropenia":  "Neutropenia",
    "neutropenia":          "Neutropenia",
    "neutropenic":          "Neutropenia",
    "thrombocytopenia":     "Thrombocytopenia",
    "platelet count decrease": "Thrombocytopenia",
    "myelosuppress":        "Myelosuppression",
    "bone marrow suppress": "Myelosuppression",
    "leukopenia":           "Myelosuppression",
    "nephrotox":            "Nephrotoxicity",
    "renal tox":            "Nephrotoxicity",
    "kidney injury":        "Nephrotoxicity",
    "hypersensitiv":        "Hypersensitivity",
    "anaphyla":              "Hypersensitivity",
    "allergic reaction":    "Hypersensitivity",
}

# Kept for anything that still wants a broad relevance check (none of the
# parsing functions below use this for node identity — only canonicalize_adr
# does that).
TARGET_ADR_KEYWORDS = list(ADR_CANONICAL_MAP.keys())

# CTCAE grades for the target ADRs — keys match ADR_CANONICAL_MAP values
# exactly, so every canonical ADR node reliably gets its CTCAE metadata.
# Also the single source of truth for the 10 canonical ADR category names
# (audit reuses list(CTCAE_MAP.keys()) instead of a second hardcoded list).
CTCAE_MAP = {
    "Ototoxicity":           {"term": "Hearing impaired",                     "grades": "1-4"},
    "Cardiotoxicity":        {"term": "Left ventricular systolic dysfunction","grades": "1-4"},
    "Peripheral Neuropathy": {"term": "Peripheral sensory neuropathy",        "grades": "1-4"},
    "Mucositis":             {"term": "Mucositis oral",                       "grades": "1-5"},
    "Hepatotoxicity":        {"term": "Alanine aminotransferase increased",   "grades": "1-4"},
    "Neutropenia":           {"term": "Neutrophil count decreased",           "grades": "1-4"},
    "Thrombocytopenia":      {"term": "Platelet count decreased",             "grades": "1-4"},
    "Myelosuppression":      {"term": "Bone marrow hypocellular",             "grades": "1-4"},
    "Nephrotoxicity":        {"term": "Acute kidney injury",                  "grades": "1-4"},
    "Hypersensitivity":      {"term": "Allergic reaction",                    "grades": "1-4"},
}

# ── Helpers ─────────────────────────────────────────────────────

def make_triple(head, head_label, relation, tail, tail_label,
                source, confidence="medium", **props):
    t = {
        "head": head, "head_label": head_label,
        "relation": relation,
        "tail": tail, "tail_label": tail_label,
        "source": source, "confidence": confidence,
    }
    t.update(props)
    return t


def is_target_drug(name):
    if not name:
        return False
    return any(d in name.lower() for d in list(TARGET_DRUGS.keys()) + list(DRUG_ALIASES.keys()))


def canonicalize_drug(name):
    """Returns the canonical target drug name for any alias, or None if this
    text doesn't refer to one of our target drugs at all. Replaces raw drug
    text as the node identity so "adriamycin" and "doxorubicin" merge.
    """
    if not name:
        return None
    n = name.lower()
    for alias, canonical in DRUG_ALIASES.items():
        if alias in n:
            return canonical
    for drug in TARGET_DRUGS.keys():
        if drug in n:
            return drug
    return None


def get_drug_class(canonical_drug_name):
    return TARGET_DRUGS.get(canonical_drug_name, "Chemotherapy")


_CATEGORY_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z /]{1,30}:\s*")


def split_multi_value_field(raw):
    """PharmGKB phenotype/side-effect fields are multi-value, comma-separated,
    and each value may carry a "Category:" prefix (Side Effect:, Toxicity:,
    Other:, Efficacy:, PD:, PK:, Dosage:, etc.). Splits on comma or semicolon
    and strips any leading "Category:" label, returning clean individual terms.
    """
    if not raw or raw == "nan":
        return []
    parts = re.split(r"[;,]", raw)
    cleaned = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        p = _CATEGORY_PREFIX_RE.sub("", p).strip()
        if p:
            cleaned.append(p)
    return cleaned


def canonicalize_adr(text):
    """Maps any raw phenotype/side-effect string to one of the target ADR
    categories, or returns None if it isn't one of our targets. E.g.
    canonicalize_adr("Side Effect: Hearing Loss"), canonicalize_adr("Hearing
    impaired"), and canonicalize_adr("Ototoxicity") all return "Ototoxicity".
    """
    if not text:
        return None
    t = text.lower()
    for keyword, canonical in ADR_CANONICAL_MAP.items():
        if keyword in t:
            return canonical
    return None


def is_target_adr(text):
    """Row-relevance gate for parse_clinical_variants /
    parse_variant_pheno_annotations, where we're checking a whole raw field
    that may contain several comma-joined terms before it's split."""
    if not text:
        return False
    return any(k in text.lower() for k in TARGET_ADR_KEYWORDS)


def enrich_with_cross_referenced_mechanism(triples):
    """LINKED_TO_ADR/AFFECTS_RESPONSE_TO edges (from clinicalVariants.tsv)
    carry PharmGKB's formal evidence_level grading but no mechanism/
    description text. ASSOCIATED_WITH_ADR/PHARMACOGENOMIC_ASSOCIATION edges
    (from variantAnnotations) carry mechanism/description but no formal
    grade. When the SAME variant connects to the SAME drug/ADR through both
    source tables, this copies the mechanism text across to the graded edge,
    tagging its origin (mechanism_source) so it's traceable back to the
    annotation source rather than looking like it came from the graded edge
    itself.
    """
    mechanism_lookup = {}
    for t in triples:
        if t["relation"] in ("ASSOCIATED_WITH_ADR", "PHARMACOGENOMIC_ASSOCIATION"):
            key = (t["head"], t["tail"])
            if t.get("mechanism") or t.get("description"):
                mechanism_lookup[key] = {
                    "mechanism": t.get("mechanism", ""),
                    "description": t.get("description", ""),
                }

    enriched = 0
    for t in triples:
        if t["relation"] in ("LINKED_TO_ADR", "AFFECTS_RESPONSE_TO"):
            key = (t["head"], t["tail"])
            if key in mechanism_lookup and not t.get("description"):
                t["mechanism"] = mechanism_lookup[key]["mechanism"]
                t["description"] = mechanism_lookup[key]["description"]
                t["mechanism_source"] = "cross-referenced from PharmGKB variantAnnotations"
                enriched += 1

    print(f"  Enriched {enriched:,} clinicalVariants edges with cross-referenced mechanism text")
    return triples


# ── Source 1 — PharmGKB Genes ────────────────────────────────────

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


# ── Source 2 — PharmGKB Drugs ────────────────────────────────────

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
        nodes.append({
            "name":       canonical,
            "drug_class": get_drug_class(canonical),
            "drug_type":  str(row.get("Type", "")).strip(),
            "cpic_level": str(row.get("Top CPIC Pairs Level", "")).strip(),
            "source_terms": raw_name,
            "label":      "Drug"
        })
    print(f"  Drugs: {len(nodes):,}")
    return nodes


# ── Source 3 — PharmGKB Variants (rsID + gene symbol reference) ──

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


# ── Source 4 — PharmGKB Clinical Variants (variant -> drug links) ─

EVIDENCE_CONFIDENCE = {
    "1A": "high", "1B": "high",
    "2A": "medium", "2B": "medium",
    "3": "low", "4": "low"
}

def split_variant_list(raw):
    """PharmGKB's variant/haplotype columns can list several distinct star
    alleles evaluated together in one annotation, comma-separated (e.g.
    "CYP2D6*1, CYP2D6*2, CYP2D6*5, CYP2D6*10"). That's different from a
    diplotype like "CYP2D6*1/*4", which uses "/" and represents one person's
    actual two-allele genotype and must NOT be split. This only splits on
    comma/semicolon, leaves "/" untouched, and never strips a leading
    "word:" pattern (HGVS notation can legitimately contain a colon, e.g.
    "NM_000106.5:c.1457G>A").
    """
    if not raw or raw == "nan":
        return []
    parts = re.split(r"[;,]", raw)
    return [p.strip() for p in parts if p.strip()]


_BASELINE_CLAUSE_RE = re.compile(r"as compared to (.+?)[\.\s]*$", re.IGNORECASE)

def _extract_baseline_text(sentence):
    """FIX: PharmGKB comparison sentences follow 'SUBJECT is associated with
    EFFECT ... as compared to BASELINE'. The baseline is a reference group,
    not itself a finding — e.g. "GSTM1 null is associated with decreased
    likelihood of Ototoxicity ... as compared to GSTM1 non-null" is a claim
    about GSTM1 null, not GSTM1 non-null. Without stripping the baseline out,
    split_variant_list() (which correctly splits comma-joined allele lists)
    was also creating an edge for the baseline allele/genotype, silently
    attaching the subject's directional claim to a group the sentence
    explicitly says is different — confirmed across GSTM1, GSTT1, TPMT,
    SLCO1B1, CYP2C19 and more, spanning ASSOCIATED_WITH_ADR and
    PHARMACOGENOMIC_ASSOCIATION edges alike. Returns the baseline clause
    text, or '' if the sentence has no comparison at all (a single-arm
    finding, where every variant token in Variant/Haplotypes is a
    legitimate subject — including cross-referenced mechanism text with no
    "as compared to" clause).
    """
    if not sentence:
        return ""
    m = _BASELINE_CLAUSE_RE.search(sentence)
    return m.group(1) if m else ""


def _normalize_for_match(s):
    return re.sub(r"\s+", "", s).lower()


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


# ── Source 5 — PharmGKB Variant Annotations ──────────────────────

def parse_variant_drug_annotations(path):
    df = pd.read_csv(path, sep="\t", low_memory=False)
    nodes   = []
    triples = []

    for _, row in df.iterrows():
        variant_raw = str(row.get("Variant/Haplotypes", "")).strip()
        gene     = str(row.get("Gene",     "")).strip()
        drug     = str(row.get("Drug(s)",  "")).strip()
        sentence = str(row.get("Sentence", "")).strip()
        pheno_cat= str(row.get("Phenotype Category", "")).strip()
        pdpk     = str(row.get("PD/PK terms", "")).strip()
        direction= str(row.get("Direction of effect", "")).strip()

        if not variant_raw or variant_raw == "nan":
            continue
        if not drug or drug == "nan":
            continue
        if not is_target_drug(drug):
            continue

        mechanism = "unknown"
        if pdpk and pdpk != "nan":
            if "metabolism" in pdpk.lower() or "pk" in pdpk.lower():
                mechanism = "pharmacokinetic"
            elif "pd" in pdpk.lower() or "efficacy" in pdpk.lower():
                mechanism = "pharmacodynamic"
        if "immune" in sentence.lower() or "hypersensitivity" in sentence.lower():
            mechanism = "immune_mediated"

        # FIX: extract the baseline clause once per row, before looping
        # over the split variants below.
        baseline_text = _extract_baseline_text(sentence)

        for variant in split_variant_list(variant_raw):
            # FIX: skip creating any edge for the baseline allele/genotype —
            # it's the reference group the sentence compares AGAINST, not a
            # finding in its own right. See _extract_baseline_text above.
            if baseline_text and _normalize_for_match(variant) in _normalize_for_match(baseline_text):
                continue

            nodes.append({
                "name":      variant,
                "hgvs":      variant if variant.startswith("c.") or
                                        variant.startswith("p.") or
                                        variant.startswith("g.") else "",
                "star_allele": variant if "*" in variant else "",
                "rsid":      variant if variant.startswith("rs") else "",
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
                        "PharmGKB_variantAnnotations", "medium",
                        mechanism=mechanism,
                        phenotype_category=pheno_cat,
                        direction=direction,
                        description=sentence[:300] if sentence else "",
                        source_term=d
                    ))

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
    return nodes, triples


def parse_variant_pheno_annotations(path):
    df = pd.read_csv(path, sep="\t", low_memory=False)
    nodes   = []
    triples = []

    for _, row in df.iterrows():
        variant_raw = str(row.get("Variant/Haplotypes", "")).strip()
        drug     = str(row.get("Drug(s)",   "")).strip()
        phenotype= str(row.get("Phenotype", "")).strip()
        side_eff = str(row.get("Side effect/efficacy/other", "")).strip()
        sentence = str(row.get("Sentence",  "")).strip()
        direction= str(row.get("Direction of effect", "")).strip()

        if not variant_raw or variant_raw == "nan":
            continue

        if not is_target_drug(drug) and not is_target_adr(phenotype):
            continue

        # FIX: extract the baseline clause once per row, before looping
        # over the split variants below.
        baseline_text = _extract_baseline_text(sentence)

        for variant in split_variant_list(variant_raw):
            # FIX: skip creating any edge for the baseline allele/genotype —
            # this is the bug confirmed for GSTM1 non-null/null, GSTT1
            # non-null/null, TPMT *1, and others: the baseline was silently
            # inheriting the subject's directional claim. See
            # _extract_baseline_text above.
            if baseline_text and _normalize_for_match(variant) in _normalize_for_match(baseline_text):
                continue

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
                            "PharmGKB_variantAnnotations", "medium",
                            side_effect_type=side_eff,
                            direction=direction,
                            description=sentence[:300] if sentence else "",
                            source_term=phen
                        ))

    print(f"  Variant-phenotype annotations: {len(triples):,} edges (filtered to oncology ADRs)")
    return nodes, triples


# ── Source 6 — CPIC ───────────────────────────────────────────────

def parse_cpic(recs_path, drugs_path):
    if not os.path.exists(recs_path):
        print("  CPIC not found - skipping")
        return [], []

    drug_lookup = {}
    if os.path.exists(drugs_path):
        with open(drugs_path) as f:
            for d in json.load(f):
                drug_id   = str(d.get("drugid", "")).strip()
                drug_name = str(d.get("name",   "")).strip()
                if drug_id and drug_name:
                    drug_lookup[drug_id] = drug_name

    with open(recs_path) as f:
        data = json.load(f)

    nodes, triples = [], []
    seen = set()

    for rec in data:
        implications = rec.get("implications", {})
        if not implications:
            continue
        gene_symbol = list(implications.keys())[0].strip()
        drug_id     = str(rec.get("drugid", "")).strip()
        drug_name_raw = drug_lookup.get(drug_id, "")

        if not gene_symbol or not drug_name_raw:
            continue
        canonical_drug = canonicalize_drug(drug_name_raw)
        if not canonical_drug:
            continue

        pair = (gene_symbol, canonical_drug)
        if pair in seen:
            continue
        seen.add(pair)

        nodes.append({"name": gene_symbol, "label": "Gene"})
        nodes.append({
            "name":       canonical_drug,
            "drug_class": get_drug_class(canonical_drug),
            "label":      "Drug"
        })

        triples.append(make_triple(
            gene_symbol, "Gene", "CPIC_GUIDELINE_FOR",
            canonical_drug, "Drug",
            "CPIC", "high",
            classification=rec.get("classification", ""),
            recommendation=str(rec.get("drugrecommendation", ""))[:300]
        ))

        lookupkey = rec.get("lookupkey", {})
        for gene, phenotype in lookupkey.items():
            if phenotype:
                pheno_name = f"{gene} {phenotype}"
                nodes.append({"name": pheno_name, "label": "Phenotype"})
                triples.append(make_triple(
                    pheno_name, "Phenotype", "RECOMMENDATION_FOR",
                    canonical_drug, "Drug",
                    "CPIC", "high",
                    gene=gene_symbol
                ))

    print(f"  CPIC: {len(triples):,} edges for oncology drugs")
    return nodes, triples


# ── Source 7 — SIDER ──────────────────────────────────────────────

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

def parse_clinvar(path):
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


# ── Neo4j loader for `build` ───────────────────────────────────────

def load_into_neo4j(all_nodes, all_triples, driver):
    with driver.session() as session:
        print("  Clearing existing data in this database...")
        session.run("MATCH (n) DETACH DELETE n")

        for label in ["Gene", "Drug", "Variant", "ADR",
                       "Phenotype", "Disease", "Mechanism"]:
            session.run(
                f"CREATE CONSTRAINT IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.name IS UNIQUE"
            )
        print("  Constraints ready.")

        by_label = defaultdict(dict)
        for node in all_nodes:
            name = node.get("name", "")
            if name and name != "nan":
                label = node["label"]
                if name not in by_label[label]:
                    by_label[label][name] = node
                else:
                    existing = by_label[label][name]
                    for k, v in node.items():
                        if v and v != "nan" and (k not in existing or not existing[k]):
                            existing[k] = v

        total_nodes = 0
        for label, nodes_dict in by_label.items():
            unique = list(nodes_dict.values())
            for i in range(0, len(unique), BATCH_SIZE):
                chunk = unique[i : i + BATCH_SIZE]
                clean = [{k: v for k, v in n.items() if k != "label"} for n in chunk]
                session.run(
                    f"UNWIND $nodes AS n "
                    f"MERGE (x:{label} {{name: n.name}}) "
                    f"SET x += n",
                    nodes=clean
                )
                total_nodes += len(chunk)
            print(f"  {label:<12}: {len(unique):,} nodes")

        print(f"  Total nodes: {total_nodes:,}")

        by_pattern = defaultdict(list)
        for t in all_triples:
            key = (t["head_label"], t["relation"], t["tail_label"])
            by_pattern[key].append(t)

        total_edges = 0
        for (hl, rel, tl), items in by_pattern.items():
            for i in range(0, len(items), BATCH_SIZE):
                chunk = items[i : i + BATCH_SIZE]
                extra_props = set()
                for item in chunk:
                    for k in item:
                        if k not in ("head", "head_label", "relation",
                                     "tail", "tail_label"):
                            extra_props.add(k)
                set_clause = ", ".join([f"r.{p} = item.{p}" for p in extra_props])
                if set_clause:
                    set_clause = "SET " + set_clause

                session.run(
                    f"UNWIND $items AS item "
                    f"MATCH (a:{hl} {{name: item.head}}) "
                    f"MATCH (b:{tl} {{name: item.tail}}) "
                    f"MERGE (a)-[r:{rel}]->(b) "
                    f"{set_clause}",
                    items=chunk
                )
                total_edges += len(chunk)

        print(f"  Total edges: {total_edges:,}")


def print_build_verification(driver):
    with driver.session() as session:
        total_n = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        total_e = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

        print(f"\n{'='*50}")
        print(f"TOTAL NODES : {total_n:,}")
        print(f"TOTAL EDGES : {total_e:,}")
        print(f"{'='*50}")

        print("\nNodes by type:")
        for rec in session.run(
            "MATCH (n) RETURN labels(n)[0] AS l, count(n) AS c ORDER BY c DESC"
        ):
            print(f"  {rec['l']:<15}: {rec['c']:,}")

        print("\nEdges by type:")
        for rec in session.run(
            "MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS c ORDER BY c DESC"
        ):
            print(f"  {rec['t']:<40}: {rec['c']:,}")

        print("\nCanonical ADR node connectivity:")
        for rec in session.run(
            "MATCH (n:ADR) "
            "RETURN n.name AS name, "
            "COUNT { (n)<-[]-() } AS incoming, "
            "COUNT { (n)-[]->() } AS outgoing "
            "ORDER BY n.name"
        ):
            print(f"  {rec['name']:<25}: incoming={rec['incoming']:,}  outgoing={rec['outgoing']:,}")

        print("\nVerifying target drug-ADR coverage:")
        pairs = [
            ("cisplatin",    "Ototoxicity"),
            ("doxorubicin",  "Cardiotoxicity"),
            ("vincristine",  "Peripheral Neuropathy"),
            ("methotrexate", "Mucositis"),
            ("methotrexate", "Hepatotoxicity"),
            ("paclitaxel",   "Peripheral Neuropathy"),
        ]
        for drug, adr in pairs:
            result = session.run(
                "MATCH (d:Drug {name:$drug})-[r]->(a:ADR {name:$adr}) "
                "RETURN count(r) AS c",
                drug=drug, adr=adr
            ).single()["c"]
            status = "OK" if result > 0 else "MISSING"
            print(f"  [{status}] {drug} -> {adr}: {result} edges")


def cmd_build():
    base = DATA_DIR
    all_nodes   = []
    all_triples = []

    print("\n" + "="*50)
    print("SOURCE 1 — PharmGKB Genes")
    print("="*50)
    gene_ref_nodes = parse_genes(
        os.path.join(base, "genes", "genes.tsv"))

    print("\n" + "="*50)
    print("SOURCE 2 — PharmGKB Drugs")
    print("="*50)
    all_nodes += parse_drugs(
        os.path.join(base, "drugs", "drugs.tsv"))

    print("\n" + "="*50)
    print("SOURCE 3 — PharmGKB Variants (rsID reference)")
    print("="*50)
    variant_ref_nodes = parse_variants(
        os.path.join(base, "variants", "variants.tsv"))

    print("\n" + "="*50)
    print("SOURCE 4 — PharmGKB Clinical Variants")
    print("="*50)
    cv_nodes, cv_triples = parse_clinical_variants(
        os.path.join(base, "clinicalVariants", "clinicalVariants.tsv"))
    all_nodes   += cv_nodes
    all_triples += cv_triples

    print("\n" + "="*50)
    print("SOURCE 5 — PharmGKB Variant Annotations (HGVS + mechanism)")
    print("="*50)
    vd_nodes, vd_triples = parse_variant_drug_annotations(
        os.path.join(base, "variantAnnotations", "var_drug_ann.tsv"))
    vp_nodes, vp_triples = parse_variant_pheno_annotations(
        os.path.join(base, "variantAnnotations", "var_pheno_ann.tsv"))
    all_nodes   += vd_nodes + vp_nodes
    all_triples += vd_triples + vp_triples

    print("\n" + "="*50)
    print("SOURCE 6 — CPIC Guidelines")
    print("="*50)
    cpic_nodes, cpic_triples = parse_cpic(
        os.path.join(base, "cpic_recommendations.json"),
        os.path.join(base, "cpic_drugs.json"))
    all_nodes   += cpic_nodes
    all_triples += cpic_triples

    print("\n" + "="*50)
    print("SOURCE 7 — SIDER (MedDRA ADR terms)")
    print("="*50)
    se_nodes, se_triples = parse_sider(
        os.path.join(base, "SIDER_side_effects.tsv.gz"),
        os.path.join(base, "SIDER_drug_names.tsv"))
    all_nodes   += se_nodes
    all_triples += se_triples

    print("\n" + "="*50)
    print("SOURCE 8 — ClinVar (clinical severity)")
    print("="*50)
    cv2_nodes, cv2_triples = parse_clinvar(
        os.path.join(base, "clinvar_variant_summary.txt.gz"))
    all_nodes   += cv2_nodes
    all_triples += cv2_triples

    print("\n" + "="*50)
    print("Enriching clinicalVariants edges with cross-referenced mechanism text")
    print("="*50)
    all_triples = enrich_with_cross_referenced_mechanism(all_triples)

    # Now that every edge source has been parsed, filter the gene/variant
    # reference dumps down to only entries that are actually an endpoint of
    # at least one edge. This is what eliminates orphaned nodes without
    # touching any node that's genuinely in use.
    referenced_gene_names = set()
    referenced_variant_names = set()
    for t in all_triples:
        if t["head_label"] == "Gene":
            referenced_gene_names.add(t["head"])
        if t["tail_label"] == "Gene":
            referenced_gene_names.add(t["tail"])
        if t["head_label"] == "Variant":
            referenced_variant_names.add(t["head"])
        if t["tail_label"] == "Variant":
            referenced_variant_names.add(t["tail"])

    gene_ref_kept = [n for n in gene_ref_nodes if n["name"] in referenced_gene_names]
    variant_ref_kept = [n for n in variant_ref_nodes if n["name"] in referenced_variant_names]

    print("\n" + "="*50)
    print("FILTERING reference dictionaries to referenced-only")
    print("="*50)
    print(f"  Genes    : kept {len(gene_ref_kept):,} / {len(gene_ref_nodes):,} "
          f"({len(gene_ref_nodes) - len(gene_ref_kept):,} orphaned rows dropped)")
    print(f"  Variants : kept {len(variant_ref_kept):,} / {len(variant_ref_nodes):,} "
          f"({len(variant_ref_nodes) - len(variant_ref_kept):,} orphaned rows dropped)")

    all_nodes += gene_ref_kept + variant_ref_kept

    print(f"\nTotal nodes to load  : {len(all_nodes):,}")
    print(f"Total edges to load  : {len(all_triples):,}")

    print("\n" + "="*50)
    print("Loading into Neo4j...")
    print("="*50)
    with get_driver() as driver:
        load_into_neo4j(all_nodes, all_triples, driver)

        print("\n" + "="*50)
        print("Verification")
        print("="*50)
        print_build_verification(driver)

    print("\nDone! Try these in Neo4j browser:")
    print()
    print("// Cisplatin -> variants -> ototoxicity")
    print("MATCH (d:Drug {name:'cisplatin'})<-[:AFFECTS_RESPONSE_TO]-(v:Variant)")
    print("      -[:LINKED_TO_ADR]->(a:ADR {name:'Ototoxicity'})")
    print("RETURN d.name, v.name, a.name LIMIT 20")
    print()
    print("// Full chain: Gene -> Variant -> Drug -> ADR")
    print("MATCH (g:Gene)-[:HAS_CLINICAL_VARIANT]->(v:Variant)")
    print("      -[:AFFECTS_RESPONSE_TO]->(d:Drug {name:'cisplatin'})")
    print("RETURN g.name, v.name, d.name LIMIT 20")
    print()
    print("// CPIC guidelines for oncology drugs")
    print("MATCH (g:Gene)-[r:CPIC_GUIDELINE_FOR]->(d:Drug)")
    print("RETURN g.name, d.name, r.classification")


# ═════════════════════════════════════════════════════════════
# LOAD — rebuild the graph from kg_export/ (portable, no source data needed)
# ═════════════════════════════════════════════════════════════

def cmd_load():
    with open(os.path.join(EXPORT_DIR, "nodes.json"), encoding="utf-8") as f:
        nodes = json.load(f)
    with open(os.path.join(EXPORT_DIR, "edges.json"), encoding="utf-8") as f:
        edges = json.load(f)

    print(f"Loaded {len(nodes):,} node records and {len(edges):,} edge records from {EXPORT_DIR}/")

    with get_driver() as driver, driver.session() as session:
        print("Clearing existing data in this database...")
        session.run("MATCH (n) DETACH DELETE n")

        for label in LABELS:
            session.run(
                f"CREATE CONSTRAINT IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.name IS UNIQUE"
            )
        print("Constraints ready.")

        by_label = defaultdict(list)
        for n in nodes:
            by_label[n["label"]].append(n["properties"])

        total_nodes = 0
        for label, props_list in by_label.items():
            for i in range(0, len(props_list), BATCH_SIZE):
                chunk = props_list[i:i + BATCH_SIZE]
                session.run(
                    f"UNWIND $nodes AS n "
                    f"MERGE (x:{label} {{name: n.name}}) "
                    f"SET x += n",
                    nodes=chunk
                )
                total_nodes += len(chunk)
            print(f"  {label:<12}: {len(props_list):,} nodes")
        print(f"Total nodes: {total_nodes:,}")

        by_pattern = defaultdict(list)
        for e in edges:
            key = (e["head_label"], e["relation"], e["tail_label"])
            by_pattern[key].append(e)

        total_edges = 0
        for (hl, rel, tl), items in by_pattern.items():
            for i in range(0, len(items), BATCH_SIZE):
                chunk = items[i:i + BATCH_SIZE]
                extra_props = set()
                for item in chunk:
                    extra_props.update(item.get("properties", {}).keys())
                set_clause = ", ".join([f"r.{p} = item.properties.{p}" for p in extra_props])
                if set_clause:
                    set_clause = "SET " + set_clause

                session.run(
                    f"UNWIND $items AS item "
                    f"MATCH (a:{hl} {{name: item.head}}) "
                    f"MATCH (b:{tl} {{name: item.tail}}) "
                    f"MERGE (a)-[r:{rel}]->(b) "
                    f"{set_clause}",
                    items=chunk
                )
                total_edges += len(chunk)
        print(f"Total edges: {total_edges:,}")

    print("\nDone.")


# ═════════════════════════════════════════════════════════════
# EXPORT — dump the live graph to kg_export/
# ═════════════════════════════════════════════════════════════

def cmd_export():
    os.makedirs(EXPORT_DIR, exist_ok=True)

    all_node_records = []
    with get_driver() as driver, driver.session() as session:
        for label in LABELS:
            result = session.run(f"MATCH (n:{label}) RETURN n AS node")
            for record in result:
                node = record["node"]
                props = dict(node.items())
                all_node_records.append({
                    "label": label,
                    "properties": props,
                })
        print(f"Exported {len(all_node_records):,} nodes")

        edge_result = session.run(
            "MATCH (a)-[r]->(b) "
            "RETURN labels(a)[0] AS head_label, a.name AS head, "
            "       type(r) AS relation, properties(r) AS rel_props, "
            "       labels(b)[0] AS tail_label, b.name AS tail"
        )
        all_edge_records = []
        for record in edge_result:
            all_edge_records.append({
                "head_label": record["head_label"],
                "head":       record["head"],
                "relation":   record["relation"],
                "tail_label": record["tail_label"],
                "tail":       record["tail"],
                "properties": dict(record["rel_props"]),
            })
        print(f"Exported {len(all_edge_records):,} edges")

    with open(os.path.join(EXPORT_DIR, "nodes.json"), "w", encoding="utf-8") as f:
        json.dump(all_node_records, f, indent=2, ensure_ascii=False)

    with open(os.path.join(EXPORT_DIR, "edges.json"), "w", encoding="utf-8") as f:
        json.dump(all_edge_records, f, indent=2, ensure_ascii=False)

    nodes_size = os.path.getsize(os.path.join(EXPORT_DIR, "nodes.json")) / 1024
    edges_size = os.path.getsize(os.path.join(EXPORT_DIR, "edges.json")) / 1024
    print(f"\nWrote {EXPORT_DIR}/nodes.json ({nodes_size:.1f} KB)")
    print(f"Wrote {EXPORT_DIR}/edges.json ({edges_size:.1f} KB)")


# ═════════════════════════════════════════════════════════════
# AUDIT — health-check an existing graph
# ═════════════════════════════════════════════════════════════
#
#   1. Node counts by label
#   2. Leftover-fragmentation scan (names still containing "," "|" or a
#      "Category:" prefix — signs the splitting/canonicalization missed
#      something)
#   3. Blank/"nan" name scan
#   4. Orphaned nodes (zero relationships in any direction) per label
#   5. Case-insensitive duplicate scan for Gene (the one label with no
#      canonicalization step)
#   6. Canonical ADR connectivity (the 10 target categories)
#   7. Drug connectivity (all target drugs — flags any that loaded as a
#      node but never got an edge)
#   8. END-TO-END reasoning chain check for the 6 primary drug-ADR pairs:
#      does Gene -> Variant -> Drug AND Variant -> ADR both exist, i.e. can
#      the graph support a genetic explanation, not just a raw drug-ADR edge?

CANONICAL_ADRS = list(CTCAE_MAP.keys())
AUDIT_TARGET_DRUGS = list(TARGET_DRUGS.keys())

PRIMARY_PAIRS = [
    ("cisplatin",    "Ototoxicity"),
    ("doxorubicin",  "Cardiotoxicity"),
    ("vincristine",  "Peripheral Neuropathy"),
    ("methotrexate", "Mucositis"),
    ("methotrexate", "Hepatotoxicity"),
    ("paclitaxel",   "Peripheral Neuropathy"),
]

GENE_TO_VARIANT_RELS   = ["HAS_CLINICAL_VARIANT", "HAS_VARIANT"]
VARIANT_TO_DRUG_RELS   = ["AFFECTS_RESPONSE_TO", "PHARMACOGENOMIC_ASSOCIATION"]
VARIANT_TO_ADR_RELS    = ["LINKED_TO_ADR", "ASSOCIATED_WITH_ADR", "CLINVAR_ASSOCIATED_ADR"]


def _section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


def cmd_audit():
    issues = []

    with get_driver() as driver, driver.session() as session:

        _section("1. NODE COUNTS BY LABEL")
        for label in LABELS:
            c = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
            print(f"  {label:<12}: {c:,}")

        _section("2. FRAGMENTATION SCAN (names still containing , | or Category:)")
        for label in LABELS:
            result = session.run(
                f"MATCH (n:{label}) "
                f"WHERE n.name CONTAINS ',' OR n.name CONTAINS '|' "
                f"   OR n.name =~ '^[A-Za-z][A-Za-z ]{{1,30}}:.*' "
                f"RETURN n.name AS name LIMIT 10"
            )
            rows = [r["name"] for r in result]
            if rows:
                issues.append(f"{label}: {len(rows)}+ nodes still show fragmentation artifacts")
                print(f"  [ISSUE] {label}: found fragmented-looking names, e.g.:")
                for name in rows:
                    print(f"           - {name}")
            else:
                print(f"  [OK] {label}: no fragmentation artifacts found")

        _section("3. BLANK / NAN NAME SCAN")
        for label in LABELS:
            result = session.run(
                f"MATCH (n:{label}) "
                f"WHERE n.name IS NULL OR n.name = '' OR toLower(n.name) = 'nan' "
                f"RETURN count(n) AS c"
            )
            c = result.single()["c"]
            if c > 0:
                issues.append(f"{label}: {c} nodes with blank/nan names")
                print(f"  [ISSUE] {label}: {c} nodes with blank/nan names")
            else:
                print(f"  [OK] {label}: none")

        _section("4. ORPHANED NODES (zero relationships in either direction)")
        for label in LABELS:
            result = session.run(
                f"MATCH (n:{label}) "
                f"WHERE COUNT {{ (n)--() }} = 0 "
                f"RETURN count(n) AS c, collect(n.name)[0..5] AS sample"
            )
            rec = result.single()
            c, sample = rec["c"], rec["sample"]
            total = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
            pct = (c / total * 100) if total else 0
            flag = "[ISSUE]" if (label in ("Drug", "ADR") and c > 0) else "[INFO]"
            if label in ("Drug", "ADR") and c > 0:
                issues.append(f"{label}: {c} orphaned nodes ({pct:.1f}%) — dead weight, e.g. {sample}")
            print(f"  {flag} {label}: {c:,} / {total:,} orphaned ({pct:.1f}%)"
                  + (f" — e.g. {sample}" if sample else ""))

        _section("5. GENE CASE-INSENSITIVE DUPLICATE SCAN")
        result = session.run(
            "MATCH (n:Gene) "
            "WITH toLower(n.name) AS lname, collect(n.name) AS variants "
            "WHERE size(variants) > 1 "
            "RETURN lname, variants LIMIT 15"
        )
        rows = list(result)
        if rows:
            issues.append(f"Gene: {len(rows)}+ case-insensitive duplicate groups found")
            print(f"  [ISSUE] Found case-variant duplicate gene nodes, e.g.:")
            for r in rows:
                print(f"           - {r['variants']}")
        else:
            print("  [OK] No case-insensitive duplicate gene names found")

        _section(f"6. CANONICAL ADR CONNECTIVITY (all {len(CANONICAL_ADRS)} target categories)")
        for adr in CANONICAL_ADRS:
            result = session.run(
                "OPTIONAL MATCH (n:ADR {name:$name}) "
                "RETURN n IS NOT NULL AS exists, "
                "       COUNT { (n)<-[]-() } AS incoming",
                name=adr
            )
            rec = result.single()
            if not rec["exists"]:
                issues.append(f"ADR '{adr}': node does not exist in the graph at all")
                print(f"  [MISSING] {adr}")
            elif rec["incoming"] == 0:
                issues.append(f"ADR '{adr}': node exists but has zero incoming edges")
                print(f"  [EMPTY]   {adr}: 0 incoming edges")
            else:
                print(f"  [OK]      {adr}: {rec['incoming']:,} incoming edges")

        _section(f"7. DRUG CONNECTIVITY (all {len(AUDIT_TARGET_DRUGS)} target drugs)")
        for drug in AUDIT_TARGET_DRUGS:
            result = session.run(
                "OPTIONAL MATCH (n:Drug {name:$name}) "
                "RETURN n IS NOT NULL AS exists, "
                "       COUNT { (n)--() } AS degree",
                name=drug
            )
            rec = result.single()
            if not rec["exists"]:
                print(f"  [ABSENT]  {drug}: no node loaded (not in your PharmGKB drugs.tsv under this name)")
            elif rec["degree"] == 0:
                issues.append(f"Drug '{drug}': node exists but has zero edges")
                print(f"  [EMPTY]   {drug}: node exists, 0 edges")
            else:
                print(f"  [OK]      {drug}: {rec['degree']:,} edges")

        _section("8. END-TO-END REASONING CHAIN — Gene->Variant->Drug AND Variant->ADR")
        for drug, adr in PRIMARY_PAIRS:
            gv_rel = "|".join(GENE_TO_VARIANT_RELS)
            vd_rel = "|".join(VARIANT_TO_DRUG_RELS)
            va_rel = "|".join(VARIANT_TO_ADR_RELS)

            result = session.run(
                f"MATCH (g:Gene)-[:{gv_rel}]->(v:Variant)-[:{vd_rel}]->(d:Drug {{name:$drug}}) "
                f"MATCH (v)-[:{va_rel}]->(a:ADR {{name:$adr}}) "
                f"RETURN count(DISTINCT v) AS chain_variants, "
                f"       count(DISTINCT g) AS chain_genes",
                drug=drug, adr=adr
            )
            rec = result.single()
            n_variants, n_genes = rec["chain_variants"], rec["chain_genes"]

            if n_variants == 0:
                issues.append(
                    f"{drug} -> {adr}: NO full Gene->Variant->Drug + Variant->ADR chain exists "
                    f"(this pair cannot get a real genetic explanation from the KG as-is)"
                )
                print(f"  [BROKEN]  {drug} -> {adr}: no complete reasoning chain")
            else:
                print(f"  [OK]      {drug} -> {adr}: {n_variants} variant(s) complete the chain, "
                      f"{n_genes} gene(s) involved")

    _section("SUMMARY")
    if not issues:
        print("  No issues found. KG looks structurally sound.")
    else:
        print(f"  {len(issues)} issue(s) found:\n")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")


# ═════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="OncologyKG build/load/export/audit CLI"
    )
    parser.add_argument(
        "command",
        choices=["build", "load", "export", "audit"],
        help="build: rebuild from raw source data in data/. "
             "load: rebuild from the committed kg_export/ snapshot. "
             "export: dump the live graph to kg_export/. "
             "audit: health-check an existing graph."
    )
    args = parser.parse_args()

    {
        "build":  cmd_build,
        "load":   cmd_load,
        "export": cmd_export,
        "audit":  cmd_audit,
    }[args.command]()


if __name__ == "__main__":
    main()