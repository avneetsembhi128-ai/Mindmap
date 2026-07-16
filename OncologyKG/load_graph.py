"""
load_graph.py

Rebuilds OncologyKG from the JSON files produced by export_graph.py — no
PharmGKB/ClinVar/SIDER source data or Neo4j dump needed. Useful for cloning
the repo on a new machine and getting the exact same graph back.

Run:
    python load_graph.py
"""

import json
import os
from collections import defaultdict
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

IN_DIR = "kg_export"

LABELS = ["Gene", "Drug", "Variant", "ADR", "Phenotype"]

BATCH_SIZE = 500


def load():
    with open(os.path.join(IN_DIR, "nodes.json"), encoding="utf-8") as f:
        nodes = json.load(f)
    with open(os.path.join(IN_DIR, "edges.json"), encoding="utf-8") as f:
        edges = json.load(f)

    print(f"Loaded {len(nodes):,} node records and {len(edges):,} edge records from {IN_DIR}/")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
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

    driver.close()
    print("\nDone.")


if __name__ == "__main__":
    load()