"""
kg.py — CLI for building/managing the Pediatric Oncology ADR Knowledge Graph
in Neo4j. Needs NEO4J_PASSWORD set in the environment.

  python kg.py build   parse raw source data in data/ and load it into Neo4j
  python kg.py load    reload from the committed kg_export/ snapshot (fast)
  python kg.py export  dump the live graph to kg_export/
  python kg.py audit   health-check an existing graph

The pipeline itself lives in sibling files: kg_constants.py (settings/lookup
tables), kg_helpers.py (shared helpers), kg_parsers.py (the 8 source
parsers), kg_loader.py (writes to Neo4j), kg_audit.py (the audit). This file
just wires them together and handles the command line.
"""

import argparse
import json
import os
from collections import defaultdict

# Settings/paths/lookup tables, and the shared helper functions
from kg_constants import DATA_DIR, EXPORT_DIR, BATCH_SIZE, LABELS, TARGET_DRUGS, CTCAE_MAP
from kg_helpers import get_driver, make_triple, split_multi_value_field, \
    enrich_with_cross_referenced_mechanism, load_drug_synonym_map
# The 8 source parsers `build` calls, one per data source
from kg_parsers import (
    parse_genes, parse_drugs, parse_variants, parse_clinical_variants,
    parse_summary_annotations, parse_variant_drug_annotations,
    parse_variant_pheno_annotations, parse_variant_fa_annotations,
    parse_pediatric_tags, parse_study_parameters, parse_sider, parse_clinvar,
)
from kg_loader import load_into_neo4j, print_build_verification
from kg_audit import cmd_audit


# ═════════════════════════════════════════════════════════════
# BUILD — rebuild the graph from raw ClinPGx/SIDER/ClinVar source data
# ═════════════════════════════════════════════════════════════

# Runs all 8 parsers, combines the results, and loads them into Neo4j
def cmd_build():
    # data/ is organized by source (clinpgx/sider/clinvar), not by file
    # type. "clinpgx" (not "pharmgkb") because ClinPGx is the merger of
    # PharmGKB, CPIC, and PharmCAT — one platform, one folder.
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
    vd_nodes, vd_triples, vd_map, vd_pmids = parse_variant_drug_annotations(
        os.path.join(clinpgx, "variantAnnotations", "var_drug_ann.tsv"),
        evidence_level_map)
    vp_nodes, vp_triples, vp_map, vp_pmids = parse_variant_pheno_annotations(
        os.path.join(clinpgx, "variantAnnotations", "var_pheno_ann.tsv"),
        evidence_level_map)
    vf_nodes, vf_triples, vf_map, vf_pmids = parse_variant_fa_annotations(
        os.path.join(clinpgx, "variantAnnotations", "var_fa_ann.tsv"))
    all_nodes   += vd_nodes + vp_nodes + vf_nodes
    all_triples += vd_triples + vp_triples + vf_triples

    # Combine the 3 files' annotation ID -> finding maps into one, so
    # parse_study_parameters (which has no variant/drug/ADR columns of its
    # own) can look up which finding each Variant Annotation ID belongs to
    annotation_map = defaultdict(list)
    for m in (vd_map, vp_map, vf_map):
        for k, v in m.items():
            annotation_map[k].extend(v)

    # Same idea, for each annotation's PMID
    pmid_map = {}
    for m in (vd_pmids, vp_pmids, vf_pmids):
        pmid_map.update(m)

    pediatric_ids = parse_pediatric_tags(
        os.path.join(clinpgx, "pediatric", "pediatric_variant_annotations.tsv"))

    st_nodes, st_triples = parse_study_parameters(
        os.path.join(clinpgx, "variantAnnotations", "study_parameters.tsv"),
        annotation_map, pediatric_ids, pmid_map)
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

    # Now that every source has been parsed, find every gene/variant name
    # that's actually used in an edge — used below to drop reference-dump
    # entries that never connect to anything (orphaned nodes)
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

    # Some variants are already known relevant (via a drug/ADR edge) but
    # never got a Gene->Variant edge, because the row that made them
    # relevant happened to have an empty Gene column. variants.tsv's Gene
    # Symbols column already has this mapping (parse_variants stashed it as
    # a node property) — this turns it into a real edge, but only for
    # variants we already know are relevant, so it doesn't pull in the
    # thousands of unrelated reference-file variants.
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

# Reads the committed kg_export/ JSON files and loads them straight into
# Neo4j — the fast path, no source data or parsing needed
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

# Reads every node and edge out of the live Neo4j graph and writes them
# to kg_export/nodes.json + edges.json, so they can be committed and
# reloaded elsewhere with `load`
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
# CLI
# ═════════════════════════════════════════════════════════════

# Reads the command-line argument (build/load/export/audit) and calls the matching function
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
