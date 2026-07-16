"""
audit_oncology_kg.py

Full health check for OncologyKG, run after build_oncology_kg.py. Prints a
structured report covering everything worth verifying before trusting the
graph for MindMap testing:

  1. Node counts by label
  2. Leftover-fragmentation scan (names still containing "," "|" or a
     "Category:" prefix — signs the splitting/canonicalization fix missed
     something)
  3. Blank/"nan" name scan
  4. Orphaned nodes (zero relationships in any direction) per label —
     dead weight nodes that can never contribute to any answer
  5. Case-insensitive duplicate scan for Gene (the one label with no
     canonicalization step — e.g. PharmGKB "TPMT" vs ClinVar "Tpmt" would
     currently be two different nodes)
  6. Canonical ADR connectivity (the 10 target categories)
  7. Drug connectivity (all 28 target drugs — flags any that loaded as a
     node but never got an edge, meaning they're not actually useful yet)
  8. END-TO-END reasoning chain check for the 6 primary drug-ADR pairs —
     this is the one that matters most: does Gene -> Variant -> Drug AND
     Variant -> ADR both exist for each pair, i.e. can the graph actually
     support a genetic explanation, not just a raw drug-ADR edge?

Run:
    python audit_oncology_kg.py
"""

from neo4j import GraphDatabase

NEO4J_URI      = "neo4j://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "oncology123"

LABELS = ["Gene", "Drug", "Variant", "ADR", "Phenotype"]

CANONICAL_ADRS = [
    "Ototoxicity", "Cardiotoxicity", "Peripheral Neuropathy", "Mucositis",
    "Hepatotoxicity", "Neutropenia", "Thrombocytopenia", "Myelosuppression",
    "Nephrotoxicity", "Hypersensitivity",
]

TARGET_DRUGS = [
    "cisplatin", "doxorubicin", "vincristine", "methotrexate", "paclitaxel",
    "carboplatin", "daunorubicin", "epirubicin", "idarubicin", "vinblastine",
    "cyclophosphamide", "ifosfamide", "busulfan", "melphalan", "etoposide",
    "irinotecan", "topotecan", "cytarabine", "mercaptopurine", "thioguanine",
    "fludarabine", "bleomycin", "dactinomycin", "asparaginase",
    "temozolomide", "dexrazoxane", "imatinib", "rituximab",
]

PRIMARY_PAIRS = [
    ("cisplatin",    "Ototoxicity"),
    ("doxorubicin",  "Cardiotoxicity"),
    ("vincristine",  "Peripheral Neuropathy"),
    ("methotrexate", "Mucositis"),
    ("methotrexate", "Hepatotoxicity"),
    ("paclitaxel",   "Peripheral Neuropathy"),
]

# Relationship types that can carry a variant->drug or gene->variant or
# variant->ADR connection, per build_oncology_kg.py's schema.
GENE_TO_VARIANT_RELS   = ["HAS_CLINICAL_VARIANT", "HAS_VARIANT"]
VARIANT_TO_DRUG_RELS   = ["AFFECTS_RESPONSE_TO", "PHARMACOGENOMIC_ASSOCIATION"]
VARIANT_TO_ADR_RELS    = ["LINKED_TO_ADR", "ASSOCIATED_WITH_ADR", "CLINVAR_ASSOCIATED_ADR"]


def section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


def run():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    issues = []

    with driver.session() as session:

        # ── 1. Node counts ──────────────────────────────────────────
        section("1. NODE COUNTS BY LABEL")
        for label in LABELS:
            c = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
            print(f"  {label:<12}: {c:,}")

        # ── 2. Leftover fragmentation scan ──────────────────────────
        section("2. FRAGMENTATION SCAN (names still containing , | or Category:)")
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

        # ── 3. Blank / "nan" name scan ──────────────────────────────
        section("3. BLANK / NAN NAME SCAN")
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

        # ── 4. Orphaned nodes ────────────────────────────────────────
        section("4. ORPHANED NODES (zero relationships in either direction)")
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

        # ── 5. Gene case-duplicate scan ──────────────────────────────
        section("5. GENE CASE-INSENSITIVE DUPLICATE SCAN")
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

        # ── 6. Canonical ADR connectivity ────────────────────────────
        section("6. CANONICAL ADR CONNECTIVITY (all 10 target categories)")
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

        # ── 7. Drug connectivity ─────────────────────────────────────
        section("7. DRUG CONNECTIVITY (all 28 target drugs)")
        for drug in TARGET_DRUGS:
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

        # ── 8. End-to-end reasoning chain for the 6 primary pairs ────
        section("8. END-TO-END REASONING CHAIN — Gene->Variant->Drug AND Variant->ADR")
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

    driver.close()

    # ── SUMMARY ────────────────────────────────────────────────────
    section("SUMMARY")
    if not issues:
        print("  No issues found. KG looks structurally sound.")
    else:
        print(f"  {len(issues)} issue(s) found:\n")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")


if __name__ == "__main__":
    run()