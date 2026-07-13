#!/bin/bash

#SBATCH --time=00:15:00      # this is 15 minutes (always start with this to test whether your script runs properly before requesting for a longer job time). Time format: D-H:M:S (e.g., 5-0:0:0 is 5 days)
#SBATCH --account=def-ester # DO NOT MODIFY THIS LINE
#SBATCH --mem=24G           # total CPU memory 
#SBATCH --nodes=1           # number of nodes requested
#SBATCH --ntasks-per-node=8 # CPU cores per node.           
#SBATCH --cpus-per-task=1   # number of cores per task (if your code does not support multi-processing, you can just request for 1)
#SBATCH --gpus=h100:1
##SBATCH --gres=gpu:p100l:1 # one 16G VRAM P100 GPU per node (uncomment this and comment the line above if you want to use P100)
#SBATCH --mail-user=aks63@sfu.ca # change this to your email
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

export OLLAMA_MODELS=/scratch/avneets/ollama_models # change this to the path to ollama models folder
export OLLAMA_HOST="127.0.0.1:11434"

# add the exact scratch directory to your PATH so Slurm can find 'ollama'
export PATH=/scratch/avneets:$PATH #change this to path to whatever folder ollama_models is in

cd /home/avneets/test/mindmap/MindMap # change this to the file path to the mindmap folder

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

echo "Pulling the Qwen model..."
ollama pull qwen3:8b

echo "Starting local Neo4j server on compute node..."
export NEO4J_HOME=$HOME/neo4j-local # change to path to neo4j

# Launch Neo4j 
$NEO4J_HOME/bin/neo4j start
echo "Waiting for Neo4j to initialize..."
sleep 15

echo "Activating virtual environment..."
source ~/my_env/bin/activate

echo "Launching MindMap.py..."
python3 MindMap.py

deactivate

# Clean up
echo "Shutting down background services..."
kill $OLLAMA_PID
$NEO4J_HOME/bin/neo4j stop



