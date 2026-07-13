import os
import sys
import re
import csv
import json
import pickle
import itertools
from time import sleep
import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from openai import OpenAI

# ==========================================
# 1. API CONNECTION CONFIGURATION
# ==========================================
#BASE_URL = "https://widen-oops-sandfish.ngrok-free.dev"
#HEADERS  = {"ngrok-skip-browser-warning": "true"}
# Change your old ngrok URL to point directly to the cluster's internal loop
BASE_URL = "http://127.0.0.1:11434/v1"

# You no longer need to skip ngrok browser warnings! You can leave this blank.
HEADERS = {}

MODEL = "qwen3:8b"

client = OpenAI(
    base_url=BASE_URL,
    api_key="ollama",              # Any non-empty string works
    default_headers=HEADERS,       # Bypasses ngrok's browser warning screen
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

def prompt_extract_keyword(input_text):
    prompt = f"""You are a medical assistant. Extract clinical entities, diseases, or symptoms from the following medical text.

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

def prompt_path_finding(path_input):
    prompt = f"""There are some knowledge graph paths. They follow entity->relationship->entity format.

{path_input}

Use the knowledge graph information. Convert them to natural language sentences cleanly. Use single quotation marks for entity names and relation names. Name them as Path-based Evidence 1, Path-based Evidence 2, etc.

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

def prompt_neighbor(neighbor):
    prompt = f"""There are some knowledge graph connections. They follow entity->relationship->entity list format.

{neighbor}

Use the knowledge graph information. Convert them to natural language sentences cleanly. Use single quotation marks for entity names and relation names. Name them as Neighbor-based Evidence 1, Neighbor-based Evidence 2, etc.

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

def final_answer(question_text, response_of_KG_list_path, response_of_KG_neighbor):
    prompt = f"""You are an excellent AI doctor, and you can diagnose diseases and recommend medications based on the symptoms in the conversation.

Patient input: {question_text}

You have some medical knowledge information in the following:
### {response_of_KG_list_path}
### {response_of_KG_neighbor}

What disease does the patient have? What tests should patient take to confirm the diagnosis? What recommended medications can cure the disease? Think step by step.

Output1: The answer includes disease and tests and recommended medications.
Output2: Show me inference process as a string about extract what knowledge from which Path-based Evidence or Neighbor-based Evidence, and in the end infer what result.
Output3: Draw a decision tree graph cleanly as an ASCII layout hierarchy.

Follow this exact structure style for your reply:
Output 1:
[Your Diagnosis and Medication plan]

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
        print(f"Network hitch on final diagnosis generation, retrying... Error: {e}", flush=True)
        sleep(2)
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content

def prompt_document(question, instruction):
    prompt = f"""You are an excellent AI doctor, and you can diagnose diseases and recommend medications based on the symptoms in the conversation.

Patient input:
{question}

You have some medical knowledge information in the following:
{instruction}

What disease does the patient have? What tests should patient take to confirm the diagnosis? What recommended medications can cure the disease?"""
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in prompt_document: {e}", flush=True)
        return ""

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
            max_tokens=3,  # Strict low bound to save tokens
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
def cosine_similarity_manual(x, y):
    dot_product = np.dot(x, y.T)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    return dot_product / (norm_x[:, np.newaxis] * norm_y)

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

def find_shortest_path(start_entity_name, end_entity_name, candidate_list):
    global exist_entity
    with driver.session() as session:
        result = session.run(
            "MATCH (start_entity:Entity{name:$start_entity_name}), (end_entity:Entity{name:$end_entity_name}) "
            "MATCH p = allShortestPaths((start_entity)-[*..5]->(end_entity)) "
            "RETURN p",
            start_entity_name=start_entity_name,
            end_entity_name=end_entity_name
        )
        paths = []
        short_path = 0
        for record in result:
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

def get_entity_neighbors(entity_name: str, disease_flag) -> tuple:
    disease = []
    query = """
    MATCH (e:Entity)-[r]->(n)
    WHERE e.name = $entity_name
    RETURN type(r) AS relationship_type, collect(n.name) AS neighbor_entities
    """
    with driver.session() as session:
        result = session.run(query, entity_name=entity_name)
        neighbor_list = []
        for record in result:
            rel_type = record["relationship_type"]
            if disease_flag == 1 and rel_type == 'has_symptom':
                continue
            neighbors = record["neighbor_entities"]
            if "disease" in rel_type.replace("_", " "):
                disease.extend(neighbors)
            else:
                neighbor_list.append([
                    entity_name.replace("_", " "), 
                    rel_type.replace("_", " "), 
                    ','.join([x.replace("_", " ") for x in neighbors])
                ])
    return neighbor_list, disease

# ==========================================
# 4. MAIN PIPELINE EXECUTION
# ==========================================
if __name__ == "__main__":
    #YOUR_OPENAI_KEY = 'Add Key Here'
    #os.environ['OPENAI_API_KEY'] = YOUR_OPENAI_KEY

    # Connect to the local Neo4j instance running inside the Slurm compute node
    uri = "bolt://localhost:7687"

    # Since we disabled dbms.security.auth_enabled in the config, pass auth=None
    driver = GraphDatabase.driver(uri, auth=None)

    '''uri = f"bolt://localhost:{tunnel_port}"
    username = "neo4j"
    password = "mill-deployment-marbles"

    driver = GraphDatabase.driver(uri, auth=(username, password))'''

    
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        df = pd.read_csv('./data/chatdoctor5k/train.txt', sep='\t', header=None, names=['head', 'relation', 'tail'])
        for index, row in df.iterrows():
            query = (
                "MERGE (h:Entity { name: $head_name }) "
                "MERGE (t:Entity { name: $tail_name }) "
                "MERGE (h)-[r:`" + row['relation'] + "`]->(t)"
            )
            session.run(query, head_name=row['head'], tail_name=row['tail'])
    

    re1 = r'The extracted entities are (.*?)<END>'
    re2 = r"The extracted entity is (.*?)<END>"
    re3 = r"<CLS>(.*?)<SEP>"

    with open('output.csv', 'w', newline='', encoding='utf-8') as f4:
        writer = csv.writer(f4)
        writer.writerow(['Question', 'Label', 'MindMap'])

    with open('./data/chatdoctor5k/entity_embeddings.pkl', 'rb') as f1:
        entity_embeddings = pickle.load(f1)
        
    with open('./data/chatdoctor5k/keyword_embeddings.pkl', 'rb') as f2:
        keyword_embeddings = pickle.load(f2)

    with open("./data/chatdoctor5k/NER_chatgpt.json", "r") as f:
        for line in f.readlines()[30:]:
            x = json.loads(line)
            input_str = x["qustion_output"].replace("\n", "").replace("<OOS>", "<EOS>").replace(":", "") + "<END>"
            input_text = re.findall(re3, input_str)
            
            if not input_text:
                continue
            print('\nQuestion:\n', input_text[0], flush=True)

            output_str = x["answer_output"].replace("\n", "").replace("<OOS>", "<EOS>").replace(":", "") + "<END>"
            output_text = re.findall(re3, output_str)
                 
            question_kg = re.findall(re1, input_str)
            if len(question_kg) == 0:
                question_kg = re.findall(re2, input_str)
                if len(question_kg) == 0:
                    print("<Warning> no entities found", input_str, flush=True)
                    continue
            question_kg = question_kg[0].replace("<END>", "").replace("<EOS>", "").replace("\n", "").split(", ")

            match_kg = []
            entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])

            for kg_entity in question_kg:
                if kg_entity not in keyword_embeddings["keywords"]:
                    continue
                keyword_index = keyword_embeddings["keywords"].index(kg_entity)
                kg_entity_emb = np.array(keyword_embeddings["embeddings"][keyword_index])

                cos_similarities = cosine_similarity_manual(entity_embeddings_emb, kg_entity_emb)[0]
                max_index = cos_similarities.argmax()
                          
                match_kg_i = entity_embeddings["entities"][max_index]
                while match_kg_i.replace(" ", "_") in match_kg:
                    cos_similarities[max_index] = 0
                    max_index = cos_similarities.argmax()
                    match_kg_i = entity_embeddings["entities"][max_index]

                match_kg.append(match_kg_i.replace(" ", "_"))
            
            print('Matched Entities from Embedding Lookup:', match_kg, flush=True)

            # # 4. Neo4j Knowledge Graph Path Finding
            result_path = []
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
                        paths, exist_entity = find_shortest_path(start_entity, end_entity, candidate_entity)
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
                    result_path = result_path_list[:3]  # Enforced limit down to 3 paths maximum
                elif len(start_tmp) > 1:
                    result_path = result_path_list[:3]  # Safeguard limit balance
            
            print(f"Path Finding Complete. Found {len(result_path)} paths.", flush=True)

            # # 5. Neo4j Knowledge Graph Neighbor Entities Fetching
            neighbor_list = []
            neighbor_list_disease = []
            for match_entity in match_kg:
                disease_flag = 0
                neighbors, disease = get_entity_neighbors(match_entity, disease_flag)
                neighbor_list.extend(neighbors)

                while disease:
                    new_disease = [d_tmp for d_tmp in disease if d_tmp in match_kg]
                    target_disease = new_disease if new_disease else disease
                    for disease_entity in target_disease:
                        neighbors, disease = get_entity_neighbors(disease_entity, disease_flag=1)
                        neighbor_list_disease.extend(neighbors)
                    break
            
            if len(neighbor_list) <= 5:
                neighbor_list.extend(neighbor_list_disease)
            
            print(f"Neighbor Fetch Complete. Found {len(neighbor_list)} neighbors.", flush=True)

            # # 6. Knowledge Graph Path Based Prompt Generation
            response_of_KG_list_path = "{}"
            if result_path and isinstance(result_path, list):
                result_new_path = ["->".join(total_path_i) for total_path_i in result_path[:3]] # Enforce strict slice cap
                path_str_block = "\n".join(result_new_path)
                response_of_KG_list_path = prompt_path_finding(path_str_block)
                if is_unable_to_answer(response_of_KG_list_path):
                    response_of_KG_list_path = prompt_path_finding(path_str_block)

            # # 7. Knowledge Graph Neighbor Entities Based Prompt Generation   
            neighbor_new_list = ["->".join(neighbor_i) for neighbor_i in neighbor_list]
            neighbor_input = "\n".join(neighbor_new_list[:3]) # Enforced strict limit slice cap to 3 entries

            response_of_KG_neighbor = "{}"
            if neighbor_new_list:
                response_of_KG_neighbor = prompt_neighbor(neighbor_input)
                if is_unable_to_answer(response_of_KG_neighbor):
                    response_of_KG_neighbor = prompt_neighbor(neighbor_input)

            # # 8. Final Context-Aware Medical Answer Generation
            output_all = final_answer(input_text[0], response_of_KG_list_path, response_of_KG_neighbor)
            if is_unable_to_answer(output_all):
                output_all = final_answer(input_text[0], response_of_KG_list_path, response_of_KG_neighbor)

            print('\nMindMap Output:\n', output_all, flush=True)

            # Append the structured validation metrics metrics rows
            with open('output.csv', 'a+', newline='', encoding='utf-8') as f6:
                writer = csv.writer(f6)
                writer.writerow([input_text[0], output_text[0], output_all])
                f6.flush()
