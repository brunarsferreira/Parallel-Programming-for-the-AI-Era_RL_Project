echo "============================================================"
echo "Pendulum RL Training on Grid5000"
echo "============================================================"
echo "Date: $(date)"
echo "Job ID: $OAR_JOB_ID"

# Configuration

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/pendulum_venv"
RAY_PORT=6379

# Get nodes info
NODES=$(cat $OAR_NODEFILE | sort -u)
NODE_ARRAY=($NODES)
NUM_NODES=${#NODE_ARRAY[@]}

# The head node is THIS node (where the script runs)
HEAD_NODE=$(hostname -f)
HEAD_IP=$(hostname -i | awk '{print $1}')

# Worker nodes are all OTHER nodes
WORKER_NODES=()
for node in "${NODE_ARRAY[@]}"; do
    if [ "$node" != "$HEAD_NODE" ]; then
        WORKER_NODES+=("$node")
    fi
done

echo ""
echo "Cluster Configuration:"
echo "  - Number of nodes: $NUM_NODES"
echo "  - Head node (this node): $HEAD_NODE ($HEAD_IP)"
if [ ${#WORKER_NODES[@]} -gt 0 ]; then
    echo "  - Worker nodes:"
    for worker in "${WORKER_NODES[@]}"; do
        echo "      $worker"
    done
fi

# Activate virtual environment
cd $PROJECT_DIR
source $VENV_DIR/bin/activate

echo ""
echo "Starting Ray cluster..."

# Start Ray head node
ray start --head --port=$RAY_PORT

sleep 3

# Start Ray workers on other nodes
if [ ${#WORKER_NODES[@]} -gt 0 ]; then
    for WORKER in "${WORKER_NODES[@]}"; do
        echo "  Starting Ray on $WORKER..."
        oarsh $WORKER "source $VENV_DIR/bin/activate && ray start --address=$HEAD_IP:$RAY_PORT" &
    done
    sleep 5
fi

echo ""
echo "Ray cluster status:"
ray status

echo ""
echo "============================================================"
echo "Starting Training"
echo "============================================================"

# Calculate number of runners based on allocated cores
# Each line in OAR_NODEFILE represents one allocated core
TOTAL_CORES=$(cat $OAR_NODEFILE | wc -l)
NUM_RUNNERS=$TOTAL_CORES
echo "Using $NUM_RUNNERS EnvRunners ($TOTAL_CORES total cores across $NUM_NODES nodes)"
# Use -u for unbuffered output so we see progress in real-time
python -u train_pendulum.py --env-runners $NUM_RUNNERS --iterations 200

echo ""
echo "============================================================"
echo "Cleanup"
echo "============================================================"

# Stop Ray on all nodes
ray stop
if [ ${#WORKER_NODES[@]} -gt 0 ]; then
    for WORKER in "${WORKER_NODES[@]}"; do
        oarsh $WORKER "ray stop" 2>/dev/null || true
    done
fi

echo ""
echo "Job completed at $(date)"

