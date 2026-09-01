"""
kg_loader.py — takes the parsed nodes and edges from kg_parsers.py and
writes them into Neo4j, then prints a summary of what got loaded.
"""

from collections import defaultdict

from kg_constants import LABELS, BATCH_SIZE


# Wipes the database and loads all nodes, then all edges, in batches
def load_into_neo4j(all_nodes, all_triples, driver):
    with driver.session() as session:
        print("  Clearing existing data in this database...")
        session.run("MATCH (n) DETACH DELETE n")

        # One uniqueness constraint per node type, so MERGE below can't
        # create duplicate nodes with the same name
        for label in LABELS:
            session.run(
                f"CREATE CONSTRAINT IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.name IS UNIQUE"
            )
        print("  Constraints ready.")

        # Group nodes by type and merge duplicates (same name) into one,
        # keeping the first non-blank value seen for each property
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

        # Group edges by (start type, relation, end type), so each group
        # can be loaded with one Cypher query per batch
        by_pattern = defaultdict(list)
        for t in all_triples:
            key = (t["head_label"], t["relation"], t["tail_label"])
            by_pattern[key].append(t)

        total_edges = 0
        for (hl, rel, tl), items in by_pattern.items():
            # A plain MERGE only looks at (start node, type, end node), so
            # normally one Study can only ever back ONE edge to a given
            # Variant. SUPPORTED_BY_STUDY edges carry a for_tail property
            # (which finding this study backs) specifically so the same
            # study can support several different findings about the same
            # variant — for_tail has to go INSIDE the MERGE pattern below,
            # not just SET afterward, or those extra findings get silently
            # merged into one edge and lost. (This actually happened once —
            # do not remove for_tail from the MERGE pattern.)
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
                    # Already set by the MERGE pattern itself, so don't set
                    # it again below
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


# Prints node/edge counts and checks the 6 primary drug-ADR pairs loaded correctly
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
