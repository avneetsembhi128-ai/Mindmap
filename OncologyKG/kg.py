"""
kg.py — unified CLI for OncologyKG

Builds and manages the Pediatric Oncology ADR Knowledge Graph (Gene -> Variant
-> Drug -> ADR) in Neo4j, sourced from 3 independent resources: ClinPGx (merger 
of PharmGKB, CPIC, and PharmCAT), SIDER, and ClinVar. data/ is organized by source 
(clinpgx/,sider/, clinvar/).

Subcommands:
    python kg.py load      Rebuild the graph from kg_export/ (committed to the
                            This is the fast path to reproduce the exact graph 
                            on a new machine.
    python kg.py build     Rebuild the graph from scratch by parsing raw
                            ClinPGx/SIDER/ClinVar files in data/ (see
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
# CONNECTION AND PATHS 
# ─────────────────────────────────────────────────────────────
NEO4J_URI  = os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")
EXPORT_DIR = os.path.join(SCRIPT_DIR, "kg_export")

BATCH_SIZE = 500
LABELS = ["Gene", "Drug", "Variant", "ADR", "Study"]

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
# BUILD — rebuild the graph from raw ClinPGx/SIDER/ClinVar source data
# ═════════════════════════════════════════════════════════════
#
# Focused on these 6 drug-ADR pairs:
#   cisplatin       -> ototoxicity
#   doxorubicin     -> cardiotoxicity  (anthracycline)
#   vincristine     -> peripheral neuropathy
#   methotrexate    -> mucositis
#   methotrexate    -> hepatotoxicity
#   paclitaxel      -> peripheral neuropathy

TARGET_DRUGS = {
    "cisplatin":     "Platinum compound",
    "doxorubicin":   "Anthracycline",
    "vincristine":   "Vinca alkaloid",
    "methotrexate":  "Antimetabolite",
    "paclitaxel":    "Taxane",
    # Broadened to other standard pediatric-oncology (COG-protocol)
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

# Real brand/generic synonyms are loaded from ClinPGx's own drugs.tsv (see
# build_drug_synonym_map / load_drug_synonym_map below) instead of being
# hand-maintained here. This residual dict holds ONLY synonyms confirmed
# (by spot-check against the downloaded drugs.tsv) to be missing from
# ClinPGx's own Generic Names / Trade Names columns — real clinical
# shorthand ClinPGx itself doesn't catalog, not a stand-in for the full
# alias list this used to be. Every other previously-hardcoded alias
# (adriamycin, ara-c, cytosar, 6-mercaptopurine, 6-thioguanine, ctx,
# cytoxan, cpt-11) is now captured by the real synonym data and was
# dropped from here as redundant.
RESIDUAL_DRUG_ALIASES = {
    "vp-16":         "etoposide",         # not in drugs.tsv at all
    "vp16":          "etoposide",         # not in drugs.tsv at all
    "6-mp":          "mercaptopurine",    # drugs.tsv has "6 MP" (space, not hyphen)
    "6-tg":          "thioguanine",       # drugs.tsv has no 6-TG/6 TG form, only "TG"
}

# Populated once by load_drug_synonym_map() at the start of cmd_build(),
# from ClinPGx's own drugs.tsv Generic Names / Trade Names columns —
# {lowercased synonym: canonical drug name}. Module-level (like TARGET_DRUGS
# and CTCAE_MAP) rather than threaded as a parameter, since canonicalize_drug()
# is called from many independent parser functions across this file.
_DRUG_SYNONYM_MAP = dict(RESIDUAL_DRUG_ALIASES)
_DRUG_SYNONYM_PATTERN = None

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

# Confirmed against the real downloaded data (var_pheno_ann.tsv): plain
# substring containment on "liver" matches inside "delivery"/"deliver" (e.g.
# a digoxin PK phenotype — "...direct delivery to the surface of the
# duodenum..." — was getting canonicalized as Hepatotoxicity, and pulled an
# off-target-drug row through is_target_adr's relevance gate along with it).
# Requiring a real word boundary BEFORE each keyword closes that without
# affecting any of the deliberately partial-word stems here (e.g. "ototox"
# matching "ototoxicity", "neuropath" matching "neuropathy") — those all
# still start at a genuine word boundary in legitimate text; only a
# boundary is required on entry, not on exit, so the stems still match
# their longer real forms.
_ADR_KEYWORD_PATTERNS = {
    kw: re.compile(r"\b" + re.escape(kw)) for kw in ADR_CANONICAL_MAP
}

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
    return canonicalize_drug(name) is not None


def canonicalize_drug(name):
    """Returns the canonical target drug name for any alias, or None if this
    text doesn't refer to one of our target drugs at all. Replaces raw drug
    text as the node identity so e.g. "Platinol" and "cisplatin" merge.

    Checks _DRUG_SYNONYM_MAP (real ClinPGx brand/generic names, see
    build_drug_synonym_map) via a whole-word match, not plain substring
    containment — some real synonyms are short abbreviations (e.g. "CP",
    "TG", "MP", "AD") that would otherwise false-positive-match inside
    unrelated text. TARGET_DRUGS' own (long, specific) canonical names are
    still matched by plain substring, unchanged from before — collision
    risk there is negligible and this preserves existing matching behavior
    for things like "cisplatin-based" or "cisplatin/etoposide".
    """
    if not name:
        return None
    n = name.lower()
    if _DRUG_SYNONYM_PATTERN is not None:
        m = _DRUG_SYNONYM_PATTERN.search(n)
        if m:
            return _DRUG_SYNONYM_MAP[m.group(0)]  # n is already lowercased above
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

    Also strips stray leading/trailing double-quote characters left over
    from values like drugs.tsv's Trade Names column, where an individual
    synonym that itself contains a comma is quote-wrapped (e.g. `"CP0",
    "Camptosar", "IRINOTECAN, CPT-11"`) — splitting that raw text on comma
    is still correct (both "IRINOTECAN" and "CPT-11" are legitimate,
    independent synonyms), but without this the two straddling the
    protected comma would keep a literal quote character stuck to them
    and then never match anything real again.
    """
    if not raw or raw == "nan":
        return []
    parts = re.split(r"[;,]", raw)
    cleaned = []
    for p in parts:
        p = p.strip().strip('"').strip()
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
        if _ADR_KEYWORD_PATTERNS[keyword].search(t):
            return canonical
    return None


def is_target_adr(text):
    """Row-relevance gate for parse_clinical_variants /
    parse_variant_pheno_annotations, where we're checking a whole raw field
    that may contain several comma-joined terms before it's split."""
    if not text:
        return False
    t = text.lower()
    return any(p.search(t) for p in _ADR_KEYWORD_PATTERNS.values())


def enrich_with_cross_referenced_mechanism(triples):
    """LINKED_TO_ADR/AFFECTS_RESPONSE_TO edges (from clinicalVariants.tsv)
    carry PharmGKB's formal evidence_level grading but no mechanism/
    description text. ASSOCIATED_WITH_ADR/PHARMACOGENOMIC_ASSOCIATION edges
    (from variantAnnotations) carry mechanism/description, and — since
    Task 2's parse_summary_annotations() — a real evidence_level too,
    whenever a matching Summary Annotation exists (still absent otherwise).
    That grade is independent of what this function does: it only backfills
    LINKED_TO_ADR/AFFECTS_RESPONSE_TO's missing mechanism/description text
    when the SAME variant connects to the SAME drug/ADR through both source
    tables, tagging its origin (mechanism_source) so it's traceable back to
    the annotation source rather than looking like it came from the graded
    edge itself.
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


# ── Source 1 — ClinPGx Genes ─────────────────────────────────────

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

def build_drug_synonym_map(path):
    """Builds {lowercased synonym: canonical drug name} from ClinPGx's own
    drugs.tsv Generic Names / Trade Names columns — replaces the old
    hand-maintained DRUG_ALIASES dict (now RESIDUAL_DRUG_ALIASES, see
    above) with real brand/generic synonym data, so newly-encountered
    brand names (e.g. "Platinol" for cisplatin) resolve without needing to
    be anticipated and hardcoded in advance.

    Resolved per ROW, not per individual synonym string: each row's own
    canonical drug is decided once from its primary Name column (via the
    same substring check against TARGET_DRUGS canonicalize_drug() already
    used), and every synonym in that row inherits that one answer. This is
    deliberately NOT "re-run canonicalize_drug on each split-out synonym
    independently" — confirmed against the real file that this matters:
    thioguanine's own listed synonym "2-Amino-6-mercaptopurine" contains
    "mercaptopurine" as a substring, so resolving it independently would
    wrongly attribute it to the mercaptopurine row instead of the
    thioguanine row it actually came from. Every TARGET_DRUGS entry was
    confirmed to match exactly one drugs.tsv row (no combination-product
    Name pollution), so this row-scoped resolution is unambiguous.
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)
    synonym_map = {}
    conflicts = 0
    for _, row in df.iterrows():
        primary_name = _clean_str(row.get("Name"))
        if not primary_name:
            continue
        n = primary_name.lower()
        canonical = next((d for d in TARGET_DRUGS if d in n), None)
        if not canonical:
            continue
        synonyms = [primary_name]
        for col in ("Generic Names", "Trade Names"):
            synonyms += split_multi_value_field(str(row.get(col, "")))
        for syn in synonyms:
            key = syn.strip().lower()
            if not key:
                continue
            existing = synonym_map.get(key)
            if existing and existing != canonical:
                conflicts += 1
                continue  # keep the first-seen canonical for this synonym
            synonym_map[key] = canonical
    n_drugs = len(set(synonym_map.values()))
    print(f"  Drug synonym map: {len(synonym_map):,} synonyms resolved for "
          f"{n_drugs}/{len(TARGET_DRUGS)} target drugs"
          + (f" ({conflicts} conflicting synonym(s) skipped)" if conflicts else ""))
    return synonym_map


def load_drug_synonym_map(path):
    """Populates the module-level _DRUG_SYNONYM_MAP / _DRUG_SYNONYM_PATTERN
    that canonicalize_drug()/is_target_drug() read everywhere else in this
    file. Must run once, early in cmd_build(), before any parser that
    resolves drug names runs — including parse_drugs itself.
    """
    global _DRUG_SYNONYM_MAP, _DRUG_SYNONYM_PATTERN
    _DRUG_SYNONYM_MAP = build_drug_synonym_map(path)
    _DRUG_SYNONYM_MAP.update(RESIDUAL_DRUG_ALIASES)
    # Longest-first isn't actually required for correctness (every synonym
    # maps unambiguously to one canonical drug, checked above), but it
    # keeps the pattern's alternation trying the more specific match first.
    alternatives = sorted((re.escape(s) for s in _DRUG_SYNONYM_MAP), key=len, reverse=True)
    _DRUG_SYNONYM_PATTERN = re.compile(r"\b(?:" + "|".join(alternatives) + r")\b") if alternatives else None


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


def _clean_str(val):
    """PharmGKB TSV cells come through pandas as float NaN for blanks. Returns
    a stripped string, or None (never the literal string 'nan') so downstream
    `SET x += n` in load_into_neo4j drops the property entirely instead of
    writing 'nan' as a real value."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    return s if s and s.lower() != "nan" else None


def _num_or_none(val):
    """Same NaN problem as _clean_str, but for numeric columns (Study Cases,
    Ratio Stat, Confidence Interval Start/Stop, ...) — float('nan') parses
    successfully as a float, so pd.isna() must be checked first or every
    missing numeric cell would silently become NaN instead of absent."""
    if pd.isna(val):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


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


# ── Source 5 — ClinPGx Summary Annotations (real 1A-4 evidence grade) ──

def _level_rank(level):
    """Sort key where a lower tuple = stronger evidence. Same ordering as
    OncologyKGMM.py's _evidence_score, duplicated here (not imported) since
    kg.py has no dependency on that file and shouldn't gain one for one
    helper. PharmGKB/ClinPGx levels run 1A (strongest) through 4 (weakest)."""
    m = re.match(r"(\d+)([A-Za-z]?)", level)
    return (int(m.group(1)), m.group(2) or "A") if m else (9, "Z")


def parse_summary_annotations(annotations_path, evidence_path):
    """Task 2: PharmGKB/ClinPGx's curated 1A-4 evidence GRADE for a
    variant-drug/ADR finding — distinct from clinicalVariants.tsv's own
    (older, thinner) per-variant evidence level, and distinct from the flat
    "medium" confidence PHARMACOGENOMIC_ASSOCIATION/ASSOCIATED_WITH_ADR
    edges carried until now regardless of how strong the underlying evidence
    actually was.

    Neither summary_annotations.tsv row references a Variant Annotation ID
    directly — the join runs through summary_ann_evidence.tsv, whose
    "Evidence ID" column IS the Variant Annotation ID (confirmed directly:
    13,212 of 14,094 Evidence IDs match a real Variant Annotation ID in
    var_drug_ann.tsv/var_pheno_ann.tsv; the remainder are Guideline/Label
    Annotation rows or functional-assay evidence, different evidence types
    entirely, not a data problem). That 94% figure is global, across all of
    ClinPGx — scoped down to just the 6 primary drug-ADR pairs this project
    targets, only 29.3% (382/1,305) of in-scope annotations actually have a
    Summary Annotation (24.2%-39.3% per pair), below ClinPGx's ~48%
    database-wide coverage ceiling: pediatric oncology is a newer, less-
    curated area of their Summary Annotation program, not a join problem
    here either.

    Returns {variant_annotation_id: level_of_evidence}. When one Variant
    Annotation ID is cited as evidence by more than one Summary Annotation,
    keeps the strongest (lowest-numbered) level rather than an arbitrary one.
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

def parse_variant_drug_annotations(path, evidence_level_map=None):
    df = pd.read_csv(path, sep="\t", low_memory=False)
    nodes   = []
    triples = []
    # variant_annotation_id -> [(variant, relation, tail, tail_label), ...],
    # consumed by parse_study_parameters() to attach Study nodes.
    annotation_map = defaultdict(list)
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

        if not variant_raw or variant_raw == "nan":
            continue
        if not drug or drug == "nan":
            continue
        if not is_target_drug(drug):
            continue

        # Task 2: real PharmGKB/ClinPGx 1A-4 evidence grade where a Summary
        # Annotation covers this Variant Annotation ID, replacing the flat
        # "medium" this edge type used to carry unconditionally. Left None
        # (not defaulted to "medium" or anything else) when no Summary
        # Annotation exists — an honest "we don't have a real grade for
        # this one" rather than a fabricated one.
        evidence_level = evidence_level_map.get(annotation_id) if annotation_id else None
        confidence = EVIDENCE_CONFIDENCE.get(evidence_level) if evidence_level else None

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
                        "PharmGKB_variantAnnotations", confidence,
                        mechanism=mechanism,
                        phenotype_category=pheno_cat,
                        direction=direction,
                        description=sentence[:300] if sentence else "",
                        source_term=d,
                        variant_annotation_id=annotation_id,
                        evidence_level=evidence_level
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
    return nodes, triples, annotation_map


def parse_variant_pheno_annotations(path, evidence_level_map=None):
    df = pd.read_csv(path, sep="\t", low_memory=False)
    nodes   = []
    triples = []
    annotation_map = defaultdict(list)
    evidence_level_map = evidence_level_map or {}

    for _, row in df.iterrows():
        annotation_id = _clean_str(row.get("Variant Annotation ID"))
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

        # Task 2 — see the matching comment in parse_variant_drug_annotations.
        evidence_level = evidence_level_map.get(annotation_id) if annotation_id else None
        confidence = EVIDENCE_CONFIDENCE.get(evidence_level) if evidence_level else None

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

            # FIX: this parser previously never created a node for `variant`
            # itself — only for the ADR it's associated with. Any variant
            # name not also present in var_drug_ann.tsv, var_fa_ann.tsv,
            # clinicalVariants.tsv, or the variants.tsv reference dump (true
            # for ~8% of rows here, mostly HLA alleles and named metabolizer
            # phenotypes like "DPYD deficiency") never got a Variant node
            # anywhere, so load_into_neo4j's `MATCH (a:Variant {name:...})`
            # silently failed and the ASSOCIATED_WITH_ADR edge never loaded.
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
                            description=sentence[:300] if sentence else "",
                            source_term=phen,
                            variant_annotation_id=annotation_id,
                            evidence_level=evidence_level
                        ))
                        if annotation_id:
                            annotation_map[annotation_id].append(
                                (variant, "ASSOCIATED_WITH_ADR",
                                 canonical_adr, "ADR"))

    print(f"  Variant-phenotype annotations: {len(triples):,} edges (filtered to oncology ADRs)")
    return nodes, triples, annotation_map


def parse_variant_fa_annotations(path):
    """Functional-assay evidence — in vitro / mechanistic findings (enzyme
    activity, cell assays, recombinant protein expression) from PharmGKB's
    var_fa_ann.tsv. Distinct in kind from var_drug_ann (clinical dosing/PK/PD)
    and var_pheno_ann (clinical phenotype/ADR) above: this file carries
    Assay type / Cell type columns those don't, and its Sentence describes a
    lab finding rather than a clinical outcome. Kept as its own relation
    (HAS_FUNCTIONAL_EVIDENCE) rather than folded into PHARMACOGENOMIC_ASSOCIATION
    so mechanism narratives can cite real in vitro evidence without it being
    mistaken for a clinical association. Unlike the truncated `description`
    on the other variantAnnotations edges (sentence[:300] — a known, documented
    lossy constraint), the sentence here is kept in full: there's no established
    reason to replicate that constraint in a brand-new edge type.
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)
    nodes   = []
    triples = []
    annotation_map = defaultdict(list)

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
                        variant_annotation_id=annotation_id
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
    return nodes, triples, annotation_map


def parse_pediatric_tags(path):
    """Task 5: real, curator-assessed pediatric-population flag from
    ClinPGx's pediatric dashboard export — replaces the guesswork
    age_range regex extraction in parse_study_parameters below with an
    actual reviewed determination. Every row in this file IS pediatric-
    tagged (it's the dashboard's own filtered "pediatric" result set, not
    a table with an explicit yes/no per annotation) — confirmed directly:
    2,912 of 2,926 IDs (99.5%) match a real Variant Annotation ID in
    var_drug_ann.tsv/var_pheno_ann.tsv/var_fa_ann.tsv, and 2,887 (98.7%)
    appear directly in study_parameters.tsv's own Variant Annotation ID
    column — this joins cleanly.

    Returns the set of pediatric-tagged Variant Annotation IDs. An ID's
    absence from this set means "not in ClinPGx's curated pediatric
    subset" — which could mean assessed-as-adult OR simply never
    reviewed for pediatric relevance; those two are NOT distinguishable
    from this file alone, so parse_study_parameters treats absence as
    unknown (None), never as a fabricated False.
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)
    tagged_ids = {_clean_str(v) for v in df.get("ID", []) if _clean_str(v)}
    print(f"  Pediatric tags: {len(tagged_ids):,} pediatric-tagged Variant Annotation IDs")
    return tagged_ids


def parse_study_parameters(path, annotation_map, pediatric_ids=None):
    """Study-level evidence metadata (effect size, CI, sample size, study
    design) from PharmGKB's study_parameters.tsv — the direct fix for
    "no way to judge strength/reliability of a finding." study_parameters.tsv
    has no variant/drug/ADR columns of its own; every row is joined purely on
    Variant Annotation ID against `annotation_map`, built while parsing
    var_drug_ann/var_pheno_ann/var_fa_ann above (the three sibling files that
    actually name the variant and drug/ADR). Rows whose annotation ID isn't
    in the map (e.g. because that annotation didn't involve a target
    oncology drug/ADR and its row was filtered out upstream) are skipped —
    counted and reported rather than silently dropped.

    A single evidence edge can have multiple supporting Study nodes; that's
    intentional (surfaces genuinely conflicting studies) and falls out
    naturally here since each Study node is uniquely named by its own
    PharmGKB Study Parameters ID.
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)
    pediatric_ids = pediatric_ids or set()
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
        pediatric_tagged = True if annotation_id in pediatric_ids else None

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
            "age_range":        age_range,
            "pediatric_tagged": pediatric_tagged,
            "dosing_reported":  dosing_reported,
            "variant_annotation_id": annotation_id,
            "label":            "Study"
        })

        # Variant -> SUPPORTED_BY_STUDY -> Study, once per finding this
        # study's annotation ID supports. for_relation/for_tail record which
        # specific evidence edge (relation + tail) this Study backs, since a
        # Variant can have many outgoing evidence edges and a flat
        # Variant->Study link alone wouldn't say which one this supports.
        #
        # IMPORTANT: for_tail is load-bearing, not just descriptive metadata.
        # The same (variant, study) pair can legitimately produce MULTIPLE
        # triples here with different for_tail values — e.g. one study cited
        # as evidence for both a variant's Myelosuppression association AND
        # its Hepatotoxicity association. load_into_neo4j() folds for_tail
        # into the MERGE pattern for exactly this reason: without it, all
        # triples sharing (variant, study) collapse into a single Neo4j edge
        # and every for_tail but the last-written one is silently dropped.
        # Confirmed directly: this was losing 716 of 5,818 parsed triples
        # (97% of them genuinely different findings, not duplicate rows)
        # before the MERGE key was widened. If you touch either this
        # triple-building loop or load_into_neo4j's SUPPORTED_BY_STUDY
        # handling, keep for_tail part of the relationship identity.
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

        for label in LABELS:
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
            # SUPPORTED_BY_STUDY triples (the only ones carrying a for_tail
            # property — see parse_study_parameters) need for_tail folded
            # into the MERGE pattern itself, not just SET afterward. MERGE
            # on a relationship keys purely on (start node, type, end node)
            # unless a property is written inside the pattern — a plain
            # MERGE (a)-[r:SUPPORTED_BY_STUDY]->(b) treats "the same Study
            # backing the same Variant" as ONE edge, even when that Study is
            # legitimately cited as evidence for several DIFFERENT findings
            # (different ADRs, or different drugs) about that variant.
            # Confirmed directly against a real build: this silently
            # collapsed 716 of 5,818 parsed SUPPORTED_BY_STUDY triples (479
            # distinct (variant, study) pairs) down to whichever finding
            # happened to be written last — and 97% of those (465/479) were
            # genuinely different findings being dropped, not duplicate
            # source rows (only 14/479 pairs were true redundancy). Detected
            # generically (by presence of for_tail on every item in this
            # relation-type's batch) rather than hardcoding the relation
            # name, so any future relation type with the same "one edge per
            # (endpoints, distinguishing property)" shape gets the same
            # protection automatically.
            has_for_tail = all("for_tail" in item for item in items)

            for i in range(0, len(items), BATCH_SIZE):
                chunk = items[i : i + BATCH_SIZE]
                extra_props = set()
                for item in chunk:
                    for k in item:
                        if k not in ("head", "head_label", "relation",
                                     "tail", "tail_label"):
                            extra_props.add(k)
                if has_for_tail:
                    # Already pinned inside the MERGE pattern below — leaving
                    # it in extra_props too wouldn't be wrong (the SET value
                    # would just re-write the same value MERGE already
                    # matched on), but excluding it keeps the two clauses
                    # from doing redundant, confusing double duty on the same
                    # property.
                    extra_props.discard("for_tail")
                    merge_clause = f"MERGE (a)-[r:{rel} {{for_tail: item.for_tail}}]->(b) "
                else:
                    merge_clause = f"MERGE (a)-[r:{rel}]->(b) "

                set_clause = ", ".join([f"r.{p} = item.{p}" for p in extra_props])
                if set_clause:
                    set_clause = "SET " + set_clause

                session.run(
                    f"UNWIND $items AS item "
                    f"MATCH (a:{hl} {{name: item.head}}) "
                    f"MATCH (b:{tl} {{name: item.tail}}) "
                    f"{merge_clause}"
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
    # data/ is organized by SOURCE (clinpgx/sider/clinvar), not by file
    # type — this is what makes "3 independent sources" visible directly
    # in the folder tree. "clinpgx" (not "pharmgkb") because ClinPGx is the
    # 2024-2025 merger of PharmGKB, CPIC, and PharmCAT under one platform;
    # every file below still comes from that single underlying resource.
    clinpgx = os.path.join(DATA_DIR, "clinpgx")
    sider   = os.path.join(DATA_DIR, "sider")
    clinvar = os.path.join(DATA_DIR, "clinvar")
    all_nodes   = []
    all_triples = []

    print("\n" + "="*50)
    print("Loading real drug synonym map (ClinPGx drugs.tsv Generic/Trade Names)")
    print("="*50)
    load_drug_synonym_map(os.path.join(clinpgx, "drugs", "drugs.tsv"))

    print("\n" + "="*50)
    print("SOURCE 1 — ClinPGx Genes")
    print("="*50)
    gene_ref_nodes = parse_genes(
        os.path.join(clinpgx, "genes", "genes.tsv"))

    print("\n" + "="*50)
    print("SOURCE 2 — ClinPGx Drugs")
    print("="*50)
    all_nodes += parse_drugs(
        os.path.join(clinpgx, "drugs", "drugs.tsv"))

    print("\n" + "="*50)
    print("SOURCE 3 — ClinPGx Variants (rsID reference)")
    print("="*50)
    variant_ref_nodes = parse_variants(
        os.path.join(clinpgx, "variants", "variants.tsv"))

    print("\n" + "="*50)
    print("SOURCE 4 — ClinPGx Clinical Variants")
    print("="*50)
    cv_nodes, cv_triples = parse_clinical_variants(
        os.path.join(clinpgx, "clinicalVariants", "clinicalVariants.tsv"))
    all_nodes   += cv_nodes
    all_triples += cv_triples

    print("\n" + "="*50)
    print("SOURCE 5 — ClinPGx Summary Annotations (real 1A-4 evidence grade)")
    print("="*50)
    evidence_level_map = parse_summary_annotations(
        os.path.join(clinpgx, "summaryAnnotations", "summary_annotations.tsv"),
        os.path.join(clinpgx, "summaryAnnotations", "summary_ann_evidence.tsv"))

    print("\n" + "="*50)
    print("SOURCE 6 — ClinPGx Variant Annotations (HGVS + mechanism)")
    print("="*50)
    vd_nodes, vd_triples, vd_map = parse_variant_drug_annotations(
        os.path.join(clinpgx, "variantAnnotations", "var_drug_ann.tsv"),
        evidence_level_map)
    vp_nodes, vp_triples, vp_map = parse_variant_pheno_annotations(
        os.path.join(clinpgx, "variantAnnotations", "var_pheno_ann.tsv"),
        evidence_level_map)
    vf_nodes, vf_triples, vf_map = parse_variant_fa_annotations(
        os.path.join(clinpgx, "variantAnnotations", "var_fa_ann.tsv"))
    all_nodes   += vd_nodes + vp_nodes + vf_nodes
    all_triples += vd_triples + vp_triples + vf_triples

    # Merge the three variant_annotation_id -> finding maps before handing
    # off to parse_study_parameters, which has no variant/drug/ADR columns
    # of its own and joins purely on this shared ID.
    annotation_map = defaultdict(list)
    for m in (vd_map, vp_map, vf_map):
        for k, v in m.items():
            annotation_map[k].extend(v)

    pediatric_ids = parse_pediatric_tags(
        os.path.join(clinpgx, "pediatric", "pediatric_variant_annotations.tsv"))

    st_nodes, st_triples = parse_study_parameters(
        os.path.join(clinpgx, "variantAnnotations", "study_parameters.tsv"),
        annotation_map, pediatric_ids)
    all_nodes   += st_nodes
    all_triples += st_triples

    print("\n" + "="*50)
    print("SOURCE 7 — SIDER (MedDRA ADR terms)")
    print("="*50)
    se_nodes, se_triples = parse_sider(
        os.path.join(sider, "SIDER_side_effects.tsv.gz"),
        os.path.join(sider, "SIDER_drug_names.tsv"))
    all_nodes   += se_nodes
    all_triples += se_triples

    print("\n" + "="*50)
    print("SOURCE 8 — ClinVar (clinical severity)")
    print("="*50)
    cv2_nodes, cv2_triples = parse_clinvar(
        os.path.join(clinvar, "clinvar_variant_summary.txt.gz"))
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

    # Backfill missing Gene->Variant edges for variants that are already known
    # relevant (drug/ADR-connected via some other edge above) but never got a
    # HAS_VARIANT/HAS_CLINICAL_VARIANT edge, because the PharmGKB annotation row
    # that established their drug/ADR relevance happened to have an empty Gene
    # column (confirmed for 22 of 23 variants shared between doxorubicin and
    # Cardiotoxicity — audited as a broken Gene->Variant->Drug+ADR chain despite
    # both individually being well-connected). variants.tsv's Gene Symbols column
    # already has this mapping — parse_variants stashed it as a node property
    # above; this turns it into an edge. Scoped strictly to variants already in
    # referenced_variant_names so it can't pull in the ~5,300 reference-file
    # variants the rest of this filtering step exists to drop.
    backfill_triples = []
    for node in variant_ref_nodes:
        if node["name"] not in referenced_variant_names:
            continue
        for gene in split_multi_value_field(node.get("gene_symbols", "")):
            backfill_triples.append(make_triple(
                gene, "Gene", "HAS_VARIANT",
                node["name"], "Variant",
                "PharmGKB_variants", "medium"
            ))
    print(f"  Backfilled {len(backfill_triples):,} Gene->Variant edges from "
          f"variants.tsv's Gene Symbols column")
    all_triples += backfill_triples
    for t in backfill_triples:
        referenced_gene_names.add(t["head"])

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

        _section("9. STUDY-LEVEL EVIDENCE COVERAGE (effect size / CI / n / design)")
        print("  For each chain variant: does it have a *complete* Study record (all")
        print("  of study_type/effect_size/ci_lower/n present), a Study link that's")
        print("  missing some of those fields (a PharmGKB source-data gap, not a join")
        print("  failure), or no Study link at all (only reachable via clinicalVariants.tsv,")
        print("  which carries no Variant Annotation ID to join a Study through)?")
        for drug, adr in PRIMARY_PAIRS:
            gv_rel = "|".join(GENE_TO_VARIANT_RELS)
            vd_rel = "|".join(VARIANT_TO_DRUG_RELS)
            va_rel = "|".join(VARIANT_TO_ADR_RELS)

            result = session.run(
                f"MATCH (g:Gene)-[:{gv_rel}]->(v:Variant)-[:{vd_rel}]->(d:Drug {{name:$drug}}) "
                f"MATCH (v)-[:{va_rel}]->(a:ADR {{name:$adr}}) "
                f"OPTIONAL MATCH (v)-[s:SUPPORTED_BY_STUDY]->(st:Study) "
                f"WHERE s IS NULL OR s.for_tail IN [$drug, $adr] "
                f"RETURN DISTINCT v.name AS variant, "
                f"  count(DISTINCT st) AS n_studies, "
                f"  collect(DISTINCT [st.study_type, st.effect_size, st.ci_lower, "
                f"                    st.n_cases, st.n_controls]) AS study_fields",
                drug=drug, adr=adr
            )
            rows = list(result)
            total = len(rows)
            if total == 0:
                print(f"  [N/A]     {drug} -> {adr}: no reasoning-chain variants to check (see section 8)")
                continue

            complete = incomplete = no_link = 0
            for r in rows:
                if r["n_studies"] == 0:
                    no_link += 1
                    continue
                is_complete = any(
                    f[0] is not None and f[1] is not None and f[2] is not None
                    and (f[3] is not None or f[4] is not None)
                    for f in r["study_fields"]
                )
                if is_complete:
                    complete += 1
                else:
                    incomplete += 1

            pct = complete / total * 100
            flag = "[OK]     " if pct >= 50 else "[PARTIAL]"
            print(f"  {flag} {drug} -> {adr}: {complete}/{total} variants ({pct:.1f}%) "
                  f"have complete study-level evidence "
                  f"[{incomplete} linked-but-incomplete, {no_link} no Study link]")

            if complete == 0:
                issues.append(
                    f"{drug} -> {adr}: 0/{total} chain variants have COMPLETE study-level "
                    f"evidence ({incomplete} linked-but-incomplete, {no_link} unlinkable)"
                )

        _section("10. PEDIATRIC POPULATION COVERAGE (curator-assessed, not the age_range guess)")
        print("  % of chain variants with at least one SUPPORTED_BY_STUDY link to a Study")
        print("  ClinPGx's own pediatric dashboard flagged — informational, not a pass/fail")
        print("  check: a low number here is a real, evidenced limitation to report honestly,")
        print("  not a bug in this KG.")
        for drug, adr in PRIMARY_PAIRS:
            gv_rel = "|".join(GENE_TO_VARIANT_RELS)
            vd_rel = "|".join(VARIANT_TO_DRUG_RELS)
            va_rel = "|".join(VARIANT_TO_ADR_RELS)

            result = session.run(
                f"MATCH (g:Gene)-[:{gv_rel}]->(v:Variant)-[:{vd_rel}]->(d:Drug {{name:$drug}}) "
                f"MATCH (v)-[:{va_rel}]->(a:ADR {{name:$adr}}) "
                f"OPTIONAL MATCH (v)-[s:SUPPORTED_BY_STUDY]->(st:Study) "
                f"WHERE st.pediatric_tagged = true AND s.for_tail IN [$drug, $adr] "
                f"RETURN count(DISTINCT v) AS n",
                drug=drug, adr=adr
            )
            pediatric_variants = result.single()["n"]

            total_result = session.run(
                f"MATCH (g:Gene)-[:{gv_rel}]->(v:Variant)-[:{vd_rel}]->(d:Drug {{name:$drug}}) "
                f"MATCH (v)-[:{va_rel}]->(a:ADR {{name:$adr}}) "
                f"RETURN count(DISTINCT v) AS n",
                drug=drug, adr=adr
            )
            total = total_result.single()["n"]

            if total == 0:
                print(f"  [N/A]  {drug} -> {adr}: no reasoning-chain variants to check (see section 8)")
                continue
            pct = pediatric_variants / total * 100
            print(f"  [INFO] {drug} -> {adr}: {pediatric_variants}/{total} variants ({pct:.1f}%) "
                  f"have pediatric-tagged study evidence")

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