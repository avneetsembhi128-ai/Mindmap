"""
build_oncology_kg.py (fixed)

Builds a focused Pediatric Oncology ADR Knowledge Graph from:
  1. PharmGKB clinicalVariants  — variant-drug-phenotype with evidence levels
  2. PharmGKB variantAnnotations — HGVS, mechanism (PK/PD), phenotype descriptions
  3. PharmGKB variants           — rsID, gene symbols
  4. PharmGKB drugs/genes        — reference nodes
  5. CPIC                        — gold standard guidelines
  6. SIDER                       — MedDRA ADR terms
  7. ClinVar                     — clinical severity

Focused on these 6 drug-ADR pairs:
  cisplatin       → ototoxicity
  doxorubicin     → cardiotoxicity  (anthracycline)
  vincristine     → peripheral neuropathy
  methotrexate    → mucositis
  methotrexate    → hepatotoxicity
  paclitaxel      → peripheral neuropathy

Node schema:
  Drug        — name, drug_class
  Gene        — hgnc_symbol, full_name
  Variant     — rsid, hgvs, star_allele, gene_symbol
  ADR         — meddra_term, ctcae_grade, ctcae_term
  Disease     — name, subtype
  Phenotype   — name, metabolizer_type
  Mechanism   — type (PK/PD/immune)

=====================================================================
FIXES vs. the original script (all marked with "# FIX:")
=====================================================================
Root cause found by inspecting the loaded graph: ADR nodes were fragmented
into many near-duplicates (e.g. "Side Effect:Hearing Loss", "Other:Hearing
Loss, Sensorineural", and giant unsplit blobs like "Acute lymphoblastic
leukemia,Anemia,Leukopenia,Mucositis,...") instead of collapsing into the 5
canonical ADR categories the project already defines (CTCAE_MAP /
TARGET_ADR_KEYWORDS). Three separate bugs caused this:

  1. parse_clinical_variants split `phenotypes_raw` on ";", but PharmGKB's
     phenotype field is comma-separated ("Category:term, Category:term..."),
     so multi-value fields with no semicolon survived as one giant string.
  2. Nothing stripped the "Category:" prefix (Side Effect:, Other:, Toxicity:,
     etc.), so the same underlying condition produced a different node per
     category tag it happened to be filed under.
  3. parse_clinvar never split ClinVar's pipe-delimited PhenotypeList at all —
     the whole "diseaseA|diseaseB|diseaseC" string became one ADR node name.

Fix: every raw phenotype/side-effect string, from every source, now goes
through canonicalize_adr(), which maps it to one of the 5 target ADR
categories (or discards it if irrelevant) using the keyword groups that were
already defined for this purpose. Raw source text is kept as a
`source_terms` property on the canonical node for traceability, but no
longer determines node identity. The same treatment (canonicalize_drug) is
applied to drug names, so aliases like "adriamycin" merge into "doxorubicin"
instead of creating a separate node.

Run:
    python build_oncology_kg.py
"""

import gzip
import json
import os
import re
import pandas as pd
from collections import defaultdict
from neo4j import GraphDatabase

# ─────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────
NEO4J_URI      = "neo4j://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "oncology123"

BASE_DIR = r"C:\Users\tulik\Documents\Research Project\Building KG\OncologyKG\data"

BATCH_SIZE = 500

# ─────────────────────────────────────────────────────────────
# TARGET DRUGS AND ADRs
# The original 6 pairs are the primary/best-curated cases (they get extra
# attention in the test questions elsewhere), but per user direction this KG
# is not limited to just those — expanded to standard pediatric-oncology
# chemotherapy agents and a broader set of clinically significant ADR
# categories. Both dicts below are the actual scope boundary: add/remove a
# drug or ADR keyword group here and every parsing function below picks it
# up automatically, since they all route through canonicalize_drug() /
# canonicalize_adr() rather than checking a hardcoded list per-function.
# ─────────────────────────────────────────────────────────────

TARGET_DRUGS = {
    # Original 6-pair drugs
    "cisplatin":     "Platinum compound",
    "doxorubicin":   "Anthracycline",
    "vincristine":   "Vinca alkaloid",
    "methotrexate":  "Antimetabolite",
    "paclitaxel":    "Taxane",
    # NEW: broadened to other standard pediatric-oncology (COG-protocol)
    # chemotherapy agents. This list is hand-curated, not derived from a
    # PharmGKB indication/ATC field (unconfirmed whether that column exists
    # in your drugs.tsv) — easy to prune or extend as a plain dict.
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

# FIX: aliases now map to the CANONICAL drug name they should merge into,
# instead of just being a second key in TARGET_DRUGS with its own drug_class.
# "adriamycin" is a brand/alt name for doxorubicin — it should never become
# its own node. Expanded with other common brand/alt names for the drugs
# above, since the same fragmentation risk applies to all of them, not just
# doxorubicin.
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

# FIX: canonical ADR categories, keyed by the same keywords TARGET_ADR_KEYWORDS
# already used for filtering — now also used to decide the canonical NODE
# NAME, not just whether to keep a row. Order matters: first match wins, so
# more specific keywords are listed first within reason.
# NEW: 5 additional categories per user direction ("slightly larger hand-
# picked list of oncology-relevant ADR categories") — myelosuppression,
# neutropenia, thrombocytopenia, nephrotoxicity, hypersensitivity — chosen
# as the other clinically significant, pharmacogenomically-studied ADR
# classes common across the expanded drug list above (e.g. asparaginase
# hypersensitivity, cisplatin/ifosfamide nephrotoxicity, myelosuppression
# from most cytotoxic agents).
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
    # NEW additions:
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
# parsing functions below use this for node identity anymore — only
# canonicalize_adr does that).
TARGET_ADR_KEYWORDS = list(ADR_CANONICAL_MAP.keys())

# CTCAE grades for our target ADRs — keys now match ADR_CANONICAL_MAP values
# exactly, so every canonical ADR node reliably gets its CTCAE metadata.
CTCAE_MAP = {
    "Ototoxicity":           {"term": "Hearing impaired",                     "grades": "1-4"},
    "Cardiotoxicity":        {"term": "Left ventricular systolic dysfunction","grades": "1-4"},
    "Peripheral Neuropathy": {"term": "Peripheral sensory neuropathy",        "grades": "1-4"},
    "Mucositis":             {"term": "Mucositis oral",                       "grades": "1-5"},
    "Hepatotoxicity":        {"term": "Alanine aminotransferase increased",   "grades": "1-4"},
    # NEW: CTCAE v5.0 terms for the 5 added categories.
    "Neutropenia":           {"term": "Neutrophil count decreased",           "grades": "1-4"},
    "Thrombocytopenia":      {"term": "Platelet count decreased",             "grades": "1-4"},
    "Myelosuppression":      {"term": "Bone marrow hypocellular",             "grades": "1-4"},
    "Nephrotoxicity":        {"term": "Acute kidney injury",                  "grades": "1-4"},
    "Hypersensitivity":      {"term": "Allergic reaction",                    "grades": "1-4"},
}

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

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
    """FIX: returns the canonical target drug name for any alias, or None if
    this text doesn't refer to one of our target drugs at all. Replaces raw
    drug text as the node identity so "adriamycin" and "doxorubicin" merge.
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
    """FIX: PharmGKB phenotype/side-effect fields are multi-value, comma-
    separated, and each value may carry a "Category:" prefix (Side Effect:,
    Toxicity:, Other:, Efficacy:, PD:, PK:, Dosage:, etc.). The original code
    only split on ";" (rarely present) and never stripped the prefix, which
    is why compound blobs and category-duplicated nodes showed up in the
    loaded graph. This splits on either delimiter and strips any leading
    "Category:" label, returning clean individual terms.
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
    """FIX: maps any raw phenotype/side-effect string to one of the 5 target
    ADR categories, or returns None if it isn't one of our targets. This is
    what actually fixes the fragmentation — canonicalize_adr("Side Effect:
    Hearing Loss") and canonicalize_adr("Hearing impaired") and
    canonicalize_adr("Ototoxicity") all return the same node name now.
    """
    if not text:
        return None
    t = text.lower()
    for keyword, canonical in ADR_CANONICAL_MAP.items():
        if keyword in t:
            return canonical
    return None


def is_target_adr(text):
    """Kept for the top-level row-relevance gate in parse_clinical_variants /
    parse_variant_pheno_annotations, where we're checking a whole raw field
    that may contain several comma-joined terms before we've split it yet."""
    if not text:
        return False
    return any(k in text.lower() for k in TARGET_ADR_KEYWORDS)


# ─────────────────────────────────────────────────────────────
# SOURCE 1 — PharmGKB Genes
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# SOURCE 2 — PharmGKB Drugs
# ─────────────────────────────────────────────────────────────

def parse_drugs(path):
    df = pd.read_csv(path, sep="\t", low_memory=False)
    nodes = []
    seen_names = set()
    for _, row in df.iterrows():
        raw_name = str(row.get("Name", "")).strip()
        if not raw_name or raw_name == "nan":
            continue
        # FIX: canonicalize instead of using the raw row text as the node
        # name, so alias rows (e.g. "Adriamycin") merge into "doxorubicin".
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


# ─────────────────────────────────────────────────────────────
# SOURCE 3 — PharmGKB Variants
# Gives us rsID and gene symbol for each variant
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# SOURCE 4 — PharmGKB Clinical Variants
# Gives us variant → drug links with evidence levels
# ─────────────────────────────────────────────────────────────

EVIDENCE_CONFIDENCE = {
    "1A": "high", "1B": "high",
    "2A": "medium", "2B": "medium",
    "3": "low", "4": "low"
}

def split_variant_list(raw):
    """FIX: PharmGKB's variant/haplotype columns can list several distinct
    star alleles evaluated together in one annotation, comma-separated
    (e.g. "CYP2D6*1, CYP2D6*2, CYP2D6*5, CYP2D6*10" — four separate alleles).
    That's different from a diplotype like "CYP2D6*1/*4", which uses "/" and
    represents one person's actual two-allele genotype and must NOT be
    split. This only splits on comma/semicolon, leaves "/" untouched, and
    (unlike split_multi_value_field) never strips a leading "word:" pattern,
    since HGVS notation can legitimately contain a colon
    (e.g. "NM_000106.5:c.1457G>A") and stripping it would corrupt real data.
    """
    if not raw or raw == "nan":
        return []
    parts = re.split(r"[;,]", raw)
    return [p.strip() for p in parts if p.strip()]


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

        # FIX: this is the variant-fragmentation fix — split the raw field
        # into its distinct alleles (comma/semicolon only, "/" diplotypes
        # left intact) instead of using the whole comma-joined blob as one
        # node name. Every triple below is now created per split variant.
        for variant in split_variant_list(variant_raw):
            nodes.append({
                "name":  variant,
                "rsid":  variant if variant.startswith("rs") else "",
                "source_terms": variant_raw if variant_raw != variant else "",
                "label": "Variant"
            })

            # Gene → HAS_CLINICAL_VARIANT → Variant
            if gene and gene != "nan":
                triples.append(make_triple(
                    gene, "Gene", "HAS_CLINICAL_VARIANT",
                    variant, "Variant",
                    "PharmGKB_clinicalVariants", conf,
                    evidence_level=ev_level
                ))

            # Variant → AFFECTS_RESPONSE_TO → Drug
            # FIX: split on comma-or-semicolon (was ";" only) and canonicalize
            # each piece instead of using raw chemical text as the node name.
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

            # Variant → LINKED_TO_ADR → ADR
            # FIX: this is the main fragmentation fix — split properly (comma OR
            # semicolon, category-prefix stripped) and canonicalize each piece
            # to one of the 5 target ADR nodes, instead of using the raw
            # (sometimes giant, comma-joined) phenotype blob as the node name.
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


# ─────────────────────────────────────────────────────────────
# SOURCE 5 — PharmGKB Variant Annotations
# ─────────────────────────────────────────────────────────────

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

        # FIX: same variant-fragmentation fix as parse_clinical_variants —
        # split comma/semicolon-joined allele lists, leave "/" diplotypes
        # intact, and create nodes/triples per split variant.
        for variant in split_variant_list(variant_raw):
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

            # Variant → PHARMACOGENOMIC_ASSOCIATION → Drug
            # FIX: canonicalize drug name (handles comma/semicolon lists and
            # aliases consistently with the rest of the script).
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

            # Gene → HAS_VARIANT → Variant
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

        # FIX: split the variant field too (same fragmentation bug as the
        # other two parsers) so the same allele name matches the Variant
        # nodes created elsewhere instead of silently failing to link up.
        for variant in split_variant_list(variant_raw):
            # FIX: this field previously went straight into the node name with
            # NO splitting at all. Now split + canonicalize like the other
            # phenotype fields.
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


# ─────────────────────────────────────────────────────────────
# SOURCE 6 — CPIC
# ─────────────────────────────────────────────────────────────

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
        # FIX: canonicalize instead of using the raw CPIC drug name directly.
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


# ─────────────────────────────────────────────────────────────
# SOURCE 7 — SIDER
# ─────────────────────────────────────────────────────────────

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

            # FIX: canonicalize the MedDRA term instead of keeping raw text
            # as the node name, so SIDER's phrasing merges with PharmGKB's
            # and ClinVar's for the same underlying ADR.
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


# ─────────────────────────────────────────────────────────────
# SOURCE 8 — ClinVar
# ─────────────────────────────────────────────────────────────

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

            # FIX: split ClinVar's pipe-delimited PhenotypeList and
            # canonicalize each entry, instead of treating the whole
            # "diseaseA|diseaseB|diseaseC" string as one ADR node.
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


# ─────────────────────────────────────────────────────────────
# NEO4J LOADER — unchanged from original
# ─────────────────────────────────────────────────────────────

def load_into_neo4j(all_nodes, all_triples, uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))

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

    driver.close()


# ─────────────────────────────────────────────────────────────
# VERIFY
# ─────────────────────────────────────────────────────────────

def verify(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
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

        # NEW: explicitly show the 5 canonical ADR nodes and their connectivity,
        # since that was the whole point of the fix — confirm they're no longer
        # fragmented.
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

    driver.close()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    base = BASE_DIR
    all_nodes   = []
    all_triples = []

    print("\n" + "="*50)
    print("SOURCE 1 — PharmGKB Genes")
    print("="*50)
    # FIX: don't fold this straight into all_nodes yet — genes.tsv is the
    # full PharmGKB gene catalog (25,041 rows), and most of it has no
    # relevance to this KG. Held separately so it can be filtered down to
    # only genes actually referenced by an edge, once all edges are known.
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
    # FIX: same deferred-filtering treatment as genes above — variants.tsv
    # is the full reference list; most rows aren't relevant to our target
    # drugs/ADRs and the ones that ARE relevant already get created directly
    # by parse_clinical_variants / parse_variant_drug_annotations, so this
    # is genuinely just extra reference metadata for variants already in use.
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

    # FIX: now that every edge source has been parsed, filter the gene/
    # variant reference dumps down to only entries that are actually an
    # endpoint of at least one edge. This is what eliminates the orphaned
    # nodes (98.6% of Gene, 74.1% of Variant in the last run) without
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
    load_into_neo4j(all_nodes, all_triples,
                    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    print("\n" + "="*50)
    print("Verification")
    print("="*50)
    verify(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    print("\nDone! Try these in Neo4j browser:")
    print()
    print("// Cisplatin → variants → ototoxicity")
    print("MATCH (d:Drug {name:'cisplatin'})<-[:AFFECTS_RESPONSE_TO]-(v:Variant)")
    print("      -[:LINKED_TO_ADR]->(a:ADR {name:'Ototoxicity'})")
    print("RETURN d.name, v.name, a.name LIMIT 20")
    print()
    print("// Full chain: Gene → Variant → Drug → ADR")
    print("MATCH (g:Gene)-[:HAS_CLINICAL_VARIANT]->(v:Variant)")
    print("      -[:AFFECTS_RESPONSE_TO]->(d:Drug {name:'cisplatin'})")
    print("RETURN g.name, v.name, d.name LIMIT 20")
    print()
    print("// CPIC guidelines for oncology drugs")
    print("MATCH (g:Gene)-[r:CPIC_GUIDELINE_FOR]->(d:Drug)")
    print("RETURN g.name, d.name, r.classification")