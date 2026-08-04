#!/bin/bash

#SBATCH --time=01:30:00      # bumped from 45 min after job 18419573 timed out mid-run with full evidence caps (15/30) on 10 questions. Time format: D-H:M:S (e.g., 5-0:0:0 is 5 days)
#SBATCH --account=def-ester # DO NOT MODIFY THIS LINE
#SBATCH --mem=24G           # total CPU memory
#SBATCH --nodes=1           # number of nodes requested
#SBATCH --ntasks-per-node=8 # CPU cores per node.
#SBATCH --cpus-per-task=1   # number of cores per task (if your code does not support multi-processing, you can just request for 1)
#SBATCH --gpus=h100:1
##SBATCH --gres=gpu:p100l:1 # one 16G VRAM P100 GPU per node (uncomment this and comment the line above if you want to use P100)
#SBATCH --mail-user=tva14@sfu.ca # change to your email
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --output=output%A.out

#module purge
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack
module load java/21

export OLLAMA_MODELS=/scratch/tvarma/ollama_models
export OLLAMA_HOST="127.0.0.1:11434"

# add the exact scratch directory to your PATH so Slurm can find 'ollama'
export PATH=/scratch/tvarma/ollama_install/bin:$PATH

cd /scratch/tvarma/Mindmap

#Start the Ollama server in the background
echo "Starting Ollama server..."
ollama serve > ollama_server.log 2>&1 &
OLLAMA_PID=$!

echo "Waiting for Ollama to wake up..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:11434/api/tags > /dev/null; then
        echo "Ollama is up!"
        break
    fi
    sleep 1
done

# Models were pre-pulled on the login node (compute nodes have no internet access),
# so only pull here if one is somehow missing from OLLAMA_MODELS.
if ollama list | grep -q "qwen3:8b"; then
    echo "qwen3:8b already present in OLLAMA_MODELS, skipping pull."
else
    echo "Pulling the Qwen model..."
    ollama pull qwen3:8b
fi

if ollama list | grep -q "nomic-embed-text"; then
    echo "nomic-embed-text already present in OLLAMA_MODELS, skipping pull."
else
    echo "Pulling the embedding model..."
    ollama pull nomic-embed-text
fi

echo "Starting local Neo4j server on compute node..."
export NEO4J_HOME=$HOME/neo4j-local

# Launch Neo4j
$NEO4J_HOME/bin/neo4j start
echo "Waiting for Neo4j to initialize..."
sleep 15

echo "Activating virtual environment..."
source /scratch/tvarma/my_env/bin/activate

# OncologyKGMM.py and OncologyKG/kg.py are env-var driven (see README.md) rather
# than hardcoded. Auth is disabled on this Neo4j instance (see neo4j.conf), which
# accepts any username/password, so this can be any value.
export NEO4J_URI="neo4j://127.0.0.1:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="alliance-job-placeholder"
export LLM_BASE_URL="http://127.0.0.1:11434/v1"
export LLM_MODEL="qwen3:8b"

echo "Loading the OncologyKG graph into Neo4j..."
cd OncologyKG
python kg.py load
cd ..

# Entity-matching embeddings (see generate_entity_embeddings.py) only need to be
# regenerated when the graph's node set actually changed since the last run —
# compare against nodes.json's mtime rather than recomputing on every job.
EMBEDDINGS_FILE="OncologyKG/kg_export/entity_embeddings.json"
NODES_FILE="OncologyKG/kg_export/nodes.json"
if [ -f "$EMBEDDINGS_FILE" ] && [ "$EMBEDDINGS_FILE" -nt "$NODES_FILE" ]; then
    echo "Entity embeddings already up to date with nodes.json, skipping regeneration."
else
    echo "Generating entity embeddings (nodes.json changed or embeddings missing)..."
    python generate_entity_embeddings.py
fi

echo "Launching OncologyKGMM.py..."
python3 OncologyKGMM.py

deactivate

# Clean up
echo "Shutting down background services..."
kill $OLLAMA_PID
$NEO4J_HOME/bin/neo4j stop
