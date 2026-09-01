"""
kg_constants.py — settings, folder paths, and lookup tables shared by the
other kg files (helpers, parsers, loader, audit).
This file has no functions, just values other files import.
"""

import os
import re

# ─────────────────────────────────────────────────────────────
# CONNECTION AND PATHS
# ─────────────────────────────────────────────────────────────

# Neo4j connection info — can be overridden with environment variables
NEO4J_URI  = os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")

# Folder paths used everywhere else in the project
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")
EXPORT_DIR = os.path.join(SCRIPT_DIR, "kg_export")

# How many records to send to Neo4j per request, and the node types (labels) in the graph
BATCH_SIZE = 500
LABELS = ["Gene", "Drug", "Variant", "ADR", "Study"]

# ═════════════════════════════════════════════════════════════
# BUILD — target scope for the graph
# ═════════════════════════════════════════════════════════════
#
# Focused on these 6 drug-ADR pairs:
#   cisplatin       -> ototoxicity
#   doxorubicin     -> cardiotoxicity  (anthracycline)
#   vincristine     -> peripheral neuropathy
#   methotrexate    -> mucositis
#   methotrexate    -> hepatotoxicity
#   paclitaxel      -> peripheral neuropathy

# The chemo drugs this project tracks, mapped to their drug class
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

# A few drug nicknames that ClinPGx's own drugs.tsv doesn't already cover
# (most brand/generic synonyms are loaded for real at runtime — see
# kg_helpers.build_drug_synonym_map — this dict only fills real gaps).
RESIDUAL_DRUG_ALIASES = {
    "vp-16":         "etoposide",         # not in drugs.tsv at all
    "vp16":          "etoposide",         # not in drugs.tsv at all
    "6-mp":          "mercaptopurine",    # drugs.tsv has "6 MP" (space, not hyphen)
    "6-tg":          "thioguanine",       # drugs.tsv has no 6-TG/6 TG form, only "TG"
}

# Maps different ways of writing a side effect to one standard ADR category
# name. Order matters — the first matching keyword wins.
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

# Just the keywords from the map above, as a plain list (kept for anything
# that wants a broad relevance check rather than a canonical ADR name)
TARGET_ADR_KEYWORDS = list(ADR_CANONICAL_MAP.keys())

# Compiled regex versions of the keywords above, requiring a real word
# boundary before each one (stops "liver" from matching inside "delivery")
_ADR_KEYWORD_PATTERNS = {
    kw: re.compile(r"\b" + re.escape(kw)) for kw in ADR_CANONICAL_MAP
}

# Clinical severity grading (CTCAE) info for each ADR category — also the
# master list of the 10 ADR category names other files reuse
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
