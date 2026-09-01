"""
kg_audit.py — health-check an existing graph:
This file contains the functions for checking quality and connectivity

  1. Node counts by label - counts how many nodes exist for each type of entity
  2. Leftover-fragmentation scan - looks for names that still contain characters that show data may not have been fully canonicalized  
  3. Blank/"nan" name scan - checks for nodes whose name is missing, empty or nan
  4. Orphaned nodes - fines nodes with no relationships to other nodes
  5. Case-insensitive duplicate scan - check if multiple gene nodes have the same name when capitalization
  6. Canonical ADR connectivity - checks if each of the ten ADR exist in the graph and have relationships 
  7. Drug connectivity - checks if every target drug exists in graph and is connected 
  8. END-TO-END reasoning chain check for the 6 primary drug-ADR pairs - checks if graph contains a path from gene -> variant -> drug -> ADR
"""

# Imports graph labels, ADR mapping, target drugs and functions  
from kg_constants import LABELS, CTCAE_MAP, TARGET_DRUGS
from kg_helpers import get_driver

# Get names of ADR categories and target drugs
CANONICAL_ADRS = list(CTCAE_MAP.keys())
AUDIT_TARGET_DRUGS = list(TARGET_DRUGS.keys())

# Inital ADR list
PRIMARY_PAIRS = [
    ("cisplatin",    "Ototoxicity"),
    ("doxorubicin",  "Cardiotoxicity"),
    ("vincristine",  "Peripheral Neuropathy"),
    ("methotrexate", "Mucositis"),
    ("methotrexate", "Hepatotoxicity"),
    ("paclitaxel",   "Peripheral Neuropathy"),
]

# Relationship types 
GENE_TO_VARIANT_RELS   = ["HAS_CLINICAL_VARIANT", "HAS_VARIANT"]
VARIANT_TO_DRUG_RELS   = ["AFFECTS_RESPONSE_TO", "PHARMACOGENOMIC_ASSOCIATION"]
VARIANT_TO_ADR_RELS    = ["LINKED_TO_ADR", "ASSOCIATED_WITH_ADR", "CLINVAR_ASSOCIATED_ADR"]


def _section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


# Main function that runs all audit checks 
def cmd_audit():
    issues = []

    # Connect to Neo4j 
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
