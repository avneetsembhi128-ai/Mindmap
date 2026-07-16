# Exports NEO4j graph to plain JSON file 

"""
export_graph.py

Exports the entire OncologyKG graph (all nodes + all relationships) to plain
JSON files that can be committed to git and reloaded on any machine via
load_graph.py, without needing the original PharmGKB/ClinVar/SIDER source
data or a Neo4j dump.

At the current graph size (2,631 nodes / 5,634 edges) this should produce
files in the low single-digit MB range — fine for a normal git repo.

Run after build_oncology_kg.py + any audit fixes:
    python export_graph.py

Produces:
    kg_export/nodes.json  — one entry per node, with its label and properties
    kg_export/edges.json  — one entry per relationship, with type + endpoint
                             names + properties
"""

import json
import os
from neo4j import GraphDatabase

NEO4J_URI      = "neo4j://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
if not NEO4J_PASSWORD:
    raise SystemExit(
        "Set the NEO4J_PASSWORD environment variable before running this script.\n"
        "PowerShell:  $env:NEO4J_PASSWORD = \"your-password-here\"\n"
        "bash:        export NEO4J_PASSWORD=\"your-password-here\""
    )

OUT_DIR = "kg_export"

LABELS = ["Gene", "Drug", "Variant", "ADR", "Phenotype"]


def export():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    os.makedirs(OUT_DIR, exist_ok=True)

    all_node_records = []
    with driver.session() as session:
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

    driver.close()

    with open(os.path.join(OUT_DIR, "nodes.json"), "w", encoding="utf-8") as f:
        json.dump(all_node_records, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUT_DIR, "edges.json"), "w", encoding="utf-8") as f:
        json.dump(all_edge_records, f, indent=2, ensure_ascii=False)

    nodes_size = os.path.getsize(os.path.join(OUT_DIR, "nodes.json")) / 1024
    edges_size = os.path.getsize(os.path.join(OUT_DIR, "edges.json")) / 1024
    print(f"\nWrote {OUT_DIR}/nodes.json ({nodes_size:.1f} KB)")
    print(f"Wrote {OUT_DIR}/edges.json ({edges_size:.1f} KB)")


if __name__ == "__main__":
    export()