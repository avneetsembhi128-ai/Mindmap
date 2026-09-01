"""
kg_helpers.py — shared helper functions used by every parser in kg_parsers.py.
This file contains the functions for connecting to Neo4j, building triples,
matching drug/ADR names to a standard name, and cleaning up messy TSV data.
"""

import os
import re

import pandas as pd
from neo4j import GraphDatabase

from kg_constants import NEO4J_URI, NEO4J_USER, TARGET_DRUGS, RESIDUAL_DRUG_ALIASES, \
    ADR_CANONICAL_MAP, _ADR_KEYWORD_PATTERNS


# Opens a connection to Neo4j using the password from the environment
def get_driver():
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise SystemExit(
            "Set the NEO4J_PASSWORD environment variable before running this script.\n"
            "PowerShell:  $env:NEO4J_PASSWORD = \"your-password-here\"\n"
            "bash:        export NEO4J_PASSWORD=\"your-password-here\""
        )
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, password))


# ── Triple construction ────────────────────────────────────────

# Builds one (head -> relation -> tail) edge as a plain dict, ready to load into Neo4j
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


# ── Drug canonicalization ──────────────────────────────────────

# Filled in at runtime by load_drug_synonym_map() from ClinPGx's real
# drug name data — {lowercased synonym: canonical drug name}
_DRUG_SYNONYM_MAP = dict(RESIDUAL_DRUG_ALIASES)
_DRUG_SYNONYM_PATTERN = None


# True if this text mentions one of our target drugs at all
def is_target_drug(name):
    return canonicalize_drug(name) is not None


def canonicalize_drug(name):
    """Returns the canonical target drug name for any alias, or None if it
    doesn't match one of our target drugs — e.g. "Platinol" and "cisplatin"
    both resolve to "cisplatin", so they merge into the same node.

    Matches short abbreviations (like "CP" or "MP") as a whole word only, so
    they don't accidentally match inside an unrelated word.
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


# Looks up which drug class a canonical drug name belongs to
def get_drug_class(canonical_drug_name):
    return TARGET_DRUGS.get(canonical_drug_name, "Chemotherapy")


def build_drug_synonym_map(path):
    """Builds {lowercased synonym: canonical drug name} from ClinPGx's own
    drugs.tsv Generic Names / Trade Names columns, so real brand names
    (like "Platinol" for cisplatin) resolve without being hardcoded.

    Each row's canonical drug is decided once (from its primary Name
    column), and every synonym listed in that same row inherits that
    answer — resolving synonyms one at a time instead would sometimes
    attribute a synonym to the wrong drug (e.g. thioguanine's synonym
    "2-Amino-6-mercaptopurine" contains the word "mercaptopurine").
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


# Fills in the module-level synonym map/pattern above — must run once,
# early in cmd_build(), before any parser calls canonicalize_drug()
def load_drug_synonym_map(path):
    global _DRUG_SYNONYM_MAP, _DRUG_SYNONYM_PATTERN
    _DRUG_SYNONYM_MAP = build_drug_synonym_map(path)
    _DRUG_SYNONYM_MAP.update(RESIDUAL_DRUG_ALIASES)
    # Longest-first isn't actually required for correctness (every synonym
    # maps unambiguously to one canonical drug, checked above), but it
    # keeps the pattern's alternation trying the more specific match first.
    alternatives = sorted((re.escape(s) for s in _DRUG_SYNONYM_MAP), key=len, reverse=True)
    _DRUG_SYNONYM_PATTERN = re.compile(r"\b(?:" + "|".join(alternatives) + r")\b") if alternatives else None


# ── Multi-value field splitting ────────────────────────────────

_CATEGORY_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z /]{1,30}:\s*")


def split_multi_value_field(raw):
    """Splits a comma/semicolon-separated PharmGKB field (like a list of
    side effects) into clean individual terms, stripping any leading
    "Category:" label (Side Effect:, Toxicity:, PD:, etc.) and stray quote
    characters left over from quoted values.
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


# ── ADR canonicalization ───────────────────────────────────────

# Maps any raw phenotype/side-effect string to one of our 10 target ADR
# category names, or None if it isn't a match at all
def canonicalize_adr(text):
    if not text:
        return None
    t = text.lower()
    for keyword, canonical in ADR_CANONICAL_MAP.items():
        if _ADR_KEYWORD_PATTERNS[keyword].search(t):
            return canonical
    return None


# True if this text mentions any target ADR keyword — used to check a
# whole raw field before it's split into individual terms
def is_target_adr(text):
    if not text:
        return False
    t = text.lower()
    return any(p.search(t) for p in _ADR_KEYWORD_PATTERNS.values())


def enrich_with_cross_referenced_mechanism(triples):
    """clinicalVariants.tsv edges (LINKED_TO_ADR/AFFECTS_RESPONSE_TO) have an
    evidence grade but no mechanism/description text. variantAnnotations
    edges (ASSOCIATED_WITH_ADR/PHARMACOGENOMIC_ASSOCIATION) do have that
    text. When the same variant connects to the same drug/ADR through both,
    this copies the mechanism/description text over to the edge that was
    missing it, and tags where it came from (mechanism_source).
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


# ── Evidence-level confidence mapping (shared by several parsers) ─

# Turns PharmGKB's 1A-4 evidence grade into a simple high/medium/low confidence
EVIDENCE_CONFIDENCE = {
    "1A": "high", "1B": "high",
    "2A": "medium", "2B": "medium",
    "3": "low", "4": "low"
}


def _level_rank(level):
    """Sort key where a lower tuple = stronger evidence. PharmGKB/ClinPGx
    levels run 1A (strongest) through 4 (weakest)."""
    m = re.match(r"(\d+)([A-Za-z]?)", level)
    return (int(m.group(1)), m.group(2) or "A") if m else (9, "Z")


# ── Variant/haplotype string handling ──────────────────────────

def split_variant_list(raw):
    """Splits a comma-separated list of star alleles (e.g. "CYP2D6*1,
    CYP2D6*2, CYP2D6*10") into separate entries. Only splits on comma/
    semicolon — a "/" (like "CYP2D6*1/*4") is one person's actual two-allele
    genotype and must stay together, not get split apart.
    """
    if not raw or raw == "nan":
        return []
    parts = re.split(r"[;,]", raw)
    return [p.strip() for p in parts if p.strip()]


_BASELINE_CLAUSE_RE = re.compile(r"as compared to (.+?)[\.\s]*$", re.IGNORECASE)


def _extract_baseline_text(sentence):
    """PharmGKB sentences read 'SUBJECT is associated with EFFECT ... as
    compared to BASELINE'. The baseline is just the reference group being
    compared against, not a finding of its own (e.g. "GSTM1 null ...
    as compared to GSTM1 non-null" is a claim about GSTM1 null, not
    non-null). Returns the baseline clause text, or '' if the sentence
    has no "as compared to" comparison at all.
    """
    if not sentence:
        return ""
    m = _BASELINE_CLAUSE_RE.search(sentence)
    return m.group(1) if m else ""


# Strips whitespace and lowercases, so two variant names can be compared
# regardless of spacing/case differences
def _normalize_for_match(s):
    return re.sub(r"\s+", "", s).lower()


# ── NaN-safe cell cleanup ───────────────────────────────────────

def _clean_str(val):
    """Turns a blank/missing TSV cell into None instead of the literal
    string "nan" (pandas represents blanks as float NaN)."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    return s if s and s.lower() != "nan" else None


def _num_or_none(val):
    """Same as _clean_str, but for numeric columns — returns None for a
    missing cell instead of the float NaN pandas would otherwise give."""
    if pd.isna(val):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
