#!/bin/bash
# Run one shard of student training, then its evaluation, on one GPU without SLURM.
#   bash scripts/run_local.sh <config> <gpu> <shard> <n_shards>
# Example on a machine with 2 GPUs (one per terminal / tmux window):
#   bash scripts/run_local.sh configs/ranks12_a441.yaml 0 0 2
#   bash scripts/run_local.sh configs/ranks12_a441.yaml 1 1 2
set -euo pipefail
CONFIG=$1; GPU=$2; SHARD=$3; NSHARDS=$4
cd "$(dirname "$0")/.."
mkdir -p logs
LOG="logs/$(basename "$CONFIG" .yaml)_gpu${GPU}_shard${SHARD}of${NSHARDS}.log"
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
{
  echo "start $(date)  gpu=$GPU shard=$SHARD/$NSHARDS"
  python scripts/run_stage.py --config "$CONFIG" --phase train_students --shard "$SHARD/$NSHARDS"
  python scripts/run_stage.py --config "$CONFIG" --phase evaluate --shard "$SHARD/$NSHARDS"
  echo "done $(date)"
} 2>&1 | tee -a "$LOG"
