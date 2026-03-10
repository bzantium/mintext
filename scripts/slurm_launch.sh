#!/bin/bash
#SBATCH --job-name=mintext
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# JAX distributed environment
export JAX_COORDINATOR_IP=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export JAX_COORDINATOR_PORT=29500
export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_NODEID

# Optional: XLA flags for multi-node GPU
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true"

# Parse config from args
CONFIG=${1:-configs/base.yml}
shift

echo "Node $NODE_RANK/$NNODES, Coordinator: $JAX_COORDINATOR_IP:$JAX_COORDINATOR_PORT"

srun python -m mintext.train --config "$CONFIG" "$@"
