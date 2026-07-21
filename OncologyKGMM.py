import os
import sys
import re
import csv
import itertools
from time import sleep
import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from openai import OpenAI

# ==========================================
# 1. API CONNECTION CONFIGURATION
# ==========================================
BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
HEADERS = {}
MODEL = os.environ.get("LLM_MODEL", "qwen3:8b")
# Limit how much neighbor evidence is retrieved from KG - prevents long prompts sent to LLM. 
NEIGHBOR_CAP_PER_ENTITY = 15
MAX_NEIGHBOR_EVIDENCE = 30
# Create the OpenAI compatible client to communicate with Ollama. 
client = OpenAI(
    base_url=BASE_URL,
    api_key="ollama",
    default_headers=HEADERS,
)

# ==========================================
# 2. CORE INFERENCE FUNCTIONS (NATIVE OPENAI) 
# ==========================================
def chat_35(prompt):
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in chat_35: {e}", flush=True)
        return ""

# CHANGED: reference's version says "extract clinical entities, diseases, or symptoms" —
# reworded for your domain (drugs / genes / variants / ADR terms) since this function is
# now actually called in main() (the reference never called it — it used pre-extracted
# entities from NER_chatgpt.json instead, which you don't have).
def prompt_extract_keyword(input_text):
    prompt = f"""You are a pharmacogenomics assistant. Extract entities from the following patient text.

ONLY extract: drug names, gene/variant names if mentioned, and adverse drug reaction (ADR) terms.

Input Text: {input_text}

Output the extracted entities as a clean, comma-separated list. Do not include introductory text.
"""
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error extracting keywords: {e}", flush=True)
        return ""

# UNCHANGED from reference — already domain-agnostic ("knowledge graph paths").
def prompt_path_finding(path_input):
    prompt = f"""There are some knowledge graph paths. They follow entity->relationship->entity format.

{path_input}

Use the knowledge graph information. Convert them to natural language sentences cleanly. Use single quotation marks for entity names and relation names. Name them as Path-based Evidence 1, Path-based Evidence 2, etc.

State only what each path literally says — do not add mechanisms, severity, or clinical explanation that isn't present in the path itself. Do not omit any path.

Output:"""
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Network hitch on path finding, retrying... Error: {e}", flush=True)
        sleep(2)
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content

# UNCHANGED from reference — already domain-agnostic ("knowledge graph connections").
def prompt_neighbor(neighbor):
    prompt = f"""There are some knowledge graph connections. They follow entity->relationship->entity list format.

{neighbor}

Use the knowledge graph information. Convert them to natural language sentences cleanly. Use single quotation marks for entity names and relation names. Name them as Neighbor-based Evidence 1, Neighbor-based Evidence 2, etc.

State only what each connection literally says — do not add mechanisms, severity, or clinical interpretation that isn't present in the data. If a connection lists multiple neighbor entities, mention all of them.

Output:"""
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Network hitch on neighbor prompting, retrying... Error: {e}", flush=True)
        sleep(2)
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content

# CHANGED: reference is "AI doctor diagnoses disease, recommends medication" — this is the
# domain swap to pharmacogenomics ADR/gene explanation. Structure (system framing, evidence
# block, Output1/2/3, exact reply format) kept identical to the reference.
def final_answer(question_text, response_of_KG_list_path, response_of_KG_neighbor):
    prompt = f"""You are an excellent clinical pharmacogenomicist, and you explain how genetic variants affect drug metabolism and cause adverse drug reactions (ADRs), based on the patient case in the conversation.

Patient input: {question_text}

You have some pharmacogenomic knowledge information retrieved from a knowledge graph, in the following:
### {response_of_KG_list_path}
### {response_of_KG_neighbor}

What genetic variants or genes could explain the patient's adverse drug reaction? What is the mechanism? Think step by step.

Keep knowledge-graph evidence and your own general pharmacology knowledge clearly separate. Never
present a gene/variant as if it came from the evidence above unless it is actually named there. If
the evidence above is empty or does not name a specific gene/variant, say so plainly rather than
filling the gap silently.

Output1a: KG-GROUNDED — gene(s)/variant(s) explicitly named in the evidence above, and the
mechanism connecting them to the ADR if the evidence supports one. If none are named, say so.
Output1b: GENERAL KNOWLEDGE (not from this KG) — any additional genes/variants/mechanisms you know
from pharmacogenomics literature that are relevant but were NOT present in the evidence above.
Label this clearly as outside knowledge.
Output2: Show me inference process as a string about extract what knowledge from which Path-based Evidence or Neighbor-based Evidence, and in the end infer what result. Use only Output1a items here.
Output3: Draw a decision tree graph cleanly as an ASCII layout hierarchy.

Follow this exact structure style for your reply:
Output 1a:
[KG-grounded genetic explanation, or explicit statement that none was found]

Output 1b:
[Additional general-knowledge context, clearly labeled as not from the KG]

Output 2:
Path-based Evidence 1('Patient'->'has'->...)->result 1

Output 3:
Patient
└── has
    └── symptoms
"""
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Network hitch on final answer generation, retrying... Error: {e}", flush=True)
        sleep(2)
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content

# UNCHANGED from reference (unused helper, kept for fidelity).
def prompt_document(question, instruction):
    prompt = f"""You are an excellent clinical pharmacogenomicist, and you explain adverse drug reactions based on the patient case in the conversation.

Patient input:
{question}

You have some pharmacogenomic knowledge information in the following:
{instruction}

What genetic variants or genes could explain the patient's adverse drug reaction?"""
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in prompt_document: {e}", flush=True)
        return ""

# UNCHANGED from reference.
def is_unable_to_answer(response_text):
    judge_prompt = (
        "You are evaluating whether a response actually answers the question, "
        "or instead avoids, declines, or fails to answer it.\n\n"
        f'Response to evaluate:\n"""\n{response_text}\n"""\n\n'
        "Rate how well this response answers the question, from 0.0 "
        '(completely fails to answer / refuses / says "I don\'t know") '
        "to 1.0 (fully and directly answers).\n"
        "Reply with ONLY the number, nothing else."
    )
    try:
        analysis = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=3,
            temperature=0.0,
        )
        raw = analysis.choices[0].message.content.strip()
        match = re.search(r"\d*\.?\d+", raw)
        if not match:
            return True
        score = float(match.group())
        return score <= 0.6
    except:
        return False

# ==========================================
# 3. GRAPH HELPER FUNCTIONS
# ==========================================
# combine_lists: UNCHANGED from reference.
def combine_lists(*lists):
    combinations = list(itertools.product(*lists))
    results = []
    for combination in combinations:
        new_combination = []
        for sublist in combination:
            if isinstance(sublist, list):
                new_combination += sublist
            else:
                new_combination.append(sublist)
        results.append(new_combination)
    return results

# CHANGED (round 2 — bug fix): same root cause as get_entity_neighbors above. The
# directed-only query "-[*..5]->" can only find a path if it happens to run the way your
# schema's edges point. Question 1 worked because 'cisplatin HAS_SIDE_EFFECT Ototoxicity'
# happens to point forward — but if match_kg ever contains two entities whose connecting
# edges run the other way (or a mix), this returns nothing even though the KG has the
# connection. Now tries directed first (cheaper, matches the original MindMap method),
# and only falls back to an undirected search if that finds nothing. Returns which mode
# worked as a third value so you can see it happen instead of it being invisible.
def _parse_path_records(records, candidate_list):
    global exist_entity
    paths = []
    short_path = 0
    for record in records:
        path = record["p"]
        entities = []
        relations = []
        for i in range(len(path.nodes)):
            node = path.nodes[i]
            entity_name = node["name"]
            entities.append(entity_name)
            if i < len(path.relationships):
                relationship = path.relationships[i]
                relations.append(relationship.type)

        path_str = ""
        for i in range(len(entities)):
            entities[i] = entities[i].replace("_", " ")
            if entities[i] in candidate_list:
                short_path = 1
                exist_entity = entities[i]
            path_str += entities[i]
            if i < len(relations):
                relations[i] = relations[i].replace("_", " ")
                path_str += "->" + relations[i] + "->"

        if short_path == 1:
            paths = [path_str]
            break
        else:
            paths.append(path_str)
            exist_entity = {}

    if len(paths) > 5:
        paths = sorted(paths, key=len)[:5]
    return paths, exist_entity


def find_shortest_path(start_entity_name, end_entity_name, candidate_list):
    global exist_entity
    with driver.session() as session:
        result = session.run(
            "MATCH (start_entity{name:$start_entity_name}), (end_entity{name:$end_entity_name}) "
            "MATCH p = allShortestPaths((start_entity)-[*..5]->(end_entity)) "
            "RETURN p",
            start_entity_name=start_entity_name,
            end_entity_name=end_entity_name
        )
        records = list(result)

    paths, exist_entity = _parse_path_records(records, candidate_list)
    if paths and paths != ['']:
        return paths, exist_entity, "directed"

    # Directed search found nothing — try again ignoring edge direction.
    with driver.session() as session:
        result = session.run(
            "MATCH (start_entity{name:$start_entity_name}), (end_entity{name:$end_entity_name}) "
            "MATCH p = allShortestPaths((start_entity)-[*..5]-(end_entity)) "
            "RETURN p",
            start_entity_name=start_entity_name,
            end_entity_name=end_entity_name
        )
        records = list(result)

    paths, exist_entity = _parse_path_records(records, candidate_list)
    if paths and paths != ['']:
        print(f"  Note: path {start_entity_name}->{end_entity_name} found only "
              f"UNDIRECTED — evidence direction in the prompt may not reflect true causality.")
        return paths, exist_entity, "undirected_fallback"

    return paths, exist_entity, "none"

# CHANGED (round 2 — bug fix, not new functionality): the outgoing-only query below was
# structurally blind to any edge where this entity is the TARGET rather than the source —
# e.g. your KG models genetic evidence as (Variant)-[:LINKED_TO_ADR]->(ADR), so querying
# neighbors of an ADR node with only "(e)-[r]->(n)" can never surface the variants that
# cause it. That silently starved the exact evidence this project needs (genetic causes of
# ADRs), so this now also queries incoming edges. Direction is preserved in the returned
# rows (outgoing rows keep entity->relation->neighbor order; incoming rows are written
# neighbor->relation->entity) so the natural-language conversion step downstream doesn't
# get the causality backwards.
def get_entity_neighbors(entity_name: str) -> list:
    outgoing_query = """
    MATCH (e)-[r]->(n)
    WHERE e.name = $entity_name
    RETURN type(r) AS relationship_type, n.name AS neighbor_name
    ORDER BY relationship_type, neighbor_name
    """
    incoming_query = """
    MATCH (n)-[r]->(e)
    WHERE e.name = $entity_name
    RETURN type(r) AS relationship_type, n.name AS neighbor_name
    ORDER BY relationship_type, neighbor_name
    """
    with driver.session() as session:
        neighbor_list = []

        result = session.run(outgoing_query, entity_name=entity_name)
        for record in result:
            rel_type = record["relationship_type"]
            neighbor_name = record["neighbor_name"]
            neighbor_list.append([
                entity_name.replace("_", " "),
                rel_type.replace("_", " "),
                neighbor_name.replace("_", " ")
            ])

        result = session.run(incoming_query, entity_name=entity_name)
        for record in result:
            rel_type = record["relationship_type"]
            neighbor_name = record["neighbor_name"]
            neighbor_list.append([
                neighbor_name.replace("_", " "),
                rel_type.replace("_", " "),
                entity_name.replace("_", " ")
            ])
    return neighbor_list

# ==========================================
# 4. MAIN PIPELINE EXECUTION
# ==========================================
if __name__ == "__main__":

    # CHANGED: your Neo4j connection (auth required), vs. reference's auth=None localhost.
    # Read from env vars, same convention as OncologyKG/kg.py, so this reproduces on
    # another machine without editing source — just set NEO4J_PASSWORD to match
    # whatever you used for `python kg.py load` / `build`.
    NEO4J_URI      = os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687")
    NEO4J_USER     = os.environ.get("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
    if not NEO4J_PASSWORD:
        raise SystemExit(
            "Set the NEO4J_PASSWORD environment variable before running this script "
            "(must match the password used to build/load the graph in OncologyKG).\n"
            "PowerShell:  $env:NEO4J_PASSWORD = \"your-password-here\"\n"
            "bash:        export NEO4J_PASSWORD=\"your-password-here\""
        )
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # CHANGED: reference's "wipe graph + reload from train.txt" block is REMOVED —
    # your KG is already built and loaded; this script should never touch that.

    # CHANGED: reference's node-label list for matching (ChatDoctor5k used one label,
    # so it didn't need this). Your KG has multiple labels to check against.
    NODE_LABELS = ["Gene", "Drug", "Variant", "ADR", "Phenotype"]

    with open('output.csv', 'w', newline='', encoding='utf-8') as f4:
        writer = csv.writer(f4)
        writer.writerow(['Question', 'MindMap'])

    # CHANGED: reference loaded questions + pre-extracted entities from the ChatDoctor5k
    # dataset files (NER_chatgpt.json, entity/keyword embeddings .pkl). You don't have
    # those, so: your own question list here, and entities are extracted live via the LLM
    # (prompt_extract_keyword) instead of read from a pre-extracted file.
    TEST_QUESTIONS = [
        "I am a child receiving cisplatin for cancer and developed hearing loss and tinnitus. What genetic variants make me susceptible to this ototoxicity?",
        "After several rounds of cisplatin, my child is showing significant hearing loss and now needs a hearing aid. Could this be genetic.",
        "What genes or variants have been linked to cisplatin ototoxicity risk in pediatric cancer patients?"
    ]

    for input_text_0 in TEST_QUESTIONS:
        print('\nQuestion:\n', input_text_0, flush=True)

        # CHANGED: live entity extraction (reference read this from NER_chatgpt.json).
        raw_entities = prompt_extract_keyword(input_text_0)
        question_kg = [e.strip() for e in raw_entities.split(",") if e.strip()]

        if not question_kg:
            print("<Warning> no entities found", input_text_0, flush=True)
            continue

        # CHANGED: reference matched entities via cosine similarity against precomputed
        # embeddings for the ChatDoctor5k dataset (entity_embeddings.pkl / keyword_
        # embeddings.pkl). You don't have those files, so this is a direct Neo4j lookup
        # instead — exact match (case-insensitive) against your node names, across your
        # five labels.
        match_kg = []
        with driver.session() as session:
            for kg_entity in question_kg:
                entity_clean = kg_entity.strip().replace("_", " ").lower()
                found_name = None
                for label in NODE_LABELS:
                    result = session.run(
                        f"MATCH (n:{label}) WHERE toLower(n.name) = toLower($name) "
                        f"RETURN n.name AS name LIMIT 1",
                        name=entity_clean
                    )
                    records = result.data()
                    if records:
                        found_name = records[0]["name"].replace(" ", "_")
                        break
                if found_name and found_name not in match_kg:
                    match_kg.append(found_name)

        print('Matched Entities from KG Lookup:', match_kg, flush=True)

        # UNCHANGED from reference — path-combination control flow.
        result_path = []
        path_directions_seen = set()
        if len(match_kg) > 1:
            start_entity = match_kg[0]
            candidate_entity = match_kg[1:]
            result_path_list = []

            while True:
                flag = 0
                paths_list = []
                while candidate_entity:
                    end_entity = candidate_entity[0]
                    candidate_entity.remove(end_entity)
                    paths, exist_entity, direction = find_shortest_path(start_entity, end_entity, candidate_entity)
                    path_directions_seen.add(direction)
                    path_list = []
                    if not paths or paths == ['']:
                        flag = 1
                        if not candidate_entity:
                            flag = 0
                            break
                        start_entity = candidate_entity[0]
                        candidate_entity.remove(start_entity)
                        break
                    else:
                        for p in paths:
                            path_list.append(p.split('->'))
                        if path_list:
                            paths_list.append(path_list)

                    if exist_entity != {}:
                        try:
                            candidate_entity.remove(exist_entity)
                        except:
                            pass
                    start_entity = end_entity

                result_path = combine_lists(*paths_list)
                if result_path:
                    result_path_list.extend(result_path)
                if flag == 1:
                    continue
                else:
                    break

            start_tmp = []
            for path_new in result_path_list:
                if not path_new:
                    continue
                if path_new[0] not in start_tmp:
                    start_tmp.append(path_new[0])

            if len(start_tmp) == 1:
                result_path = result_path_list[:3]
            elif len(start_tmp) > 1:
                result_path = result_path_list[:3]

        print(f"Path Finding Complete. Found {len(result_path)} paths. "
              f"(direction: {', '.join(sorted(path_directions_seen)) or 'n/a'})", flush=True)

        # CHANGED (round 2 — bug fix): previously neighbor_list.extend(neighbors) collected
        # everything per entity, then main sliced the COMBINED list to [:3] below — meaning
        # if cisplatin's outgoing HAS_SIDE_EFFECT list alone had 3+ entries, Ototoxicity's
        # variants (the evidence you actually care about) never made it into the prompt at
        # all. Now capped per entity at collection time instead.
        neighbor_list = []
        for match_entity in match_kg:
            neighbors = get_entity_neighbors(match_entity)
            neighbor_list.extend(neighbors[:NEIGHBOR_CAP_PER_ENTITY])

        print(f"Neighbor Fetch Complete. Found {len(neighbor_list)} neighbors.", flush=True)

        # UNCHANGED from reference.
        response_of_KG_list_path = "{}"
        if result_path and isinstance(result_path, list):
            result_new_path = ["->".join(total_path_i) for total_path_i in result_path[:3]]
            path_str_block = "\n".join(result_new_path)
            response_of_KG_list_path = prompt_path_finding(path_str_block)
            if is_unable_to_answer(response_of_KG_list_path):
                response_of_KG_list_path = prompt_path_finding(path_str_block)

        neighbor_new_list = ["->".join(neighbor_i) for neighbor_i in neighbor_list]
        # CHANGED: was neighbor_new_list[:3] — a flat cap across ALL entities combined.
        # Now each entity already contributed at most NEIGHBOR_CAP_PER_ENTITY above, so
        # this is just an overall safety cap to keep the prompt a reasonable size, not the
        # thing doing the (broken) per-entity limiting.
        neighbor_input = "\n".join(neighbor_new_list[:MAX_NEIGHBOR_EVIDENCE])

        response_of_KG_neighbor = "{}"
        if neighbor_new_list:
            response_of_KG_neighbor = prompt_neighbor(neighbor_input)
            if is_unable_to_answer(response_of_KG_neighbor):
                response_of_KG_neighbor = prompt_neighbor(neighbor_input)

        output_all = final_answer(input_text_0, response_of_KG_list_path, response_of_KG_neighbor)
        if is_unable_to_answer(output_all):
            output_all = final_answer(input_text_0, response_of_KG_list_path, response_of_KG_neighbor)

        print('\nMindMap Output:\n', output_all, flush=True)

        with open('output.csv', 'a+', newline='', encoding='utf-8') as f6:
            writer = csv.writer(f6)
            writer.writerow([input_text_0, output_all])
            f6.flush()

    driver.close()