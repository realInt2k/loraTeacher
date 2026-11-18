#!/bin/bash
# One task of a pipeline phase. Submitted by scripts/submit_slurm.py, which sets
# --array, --time, --job-name, --dependency, --output, the GPU request and the
# variables CONFIG, PHASE, KINDS, NSHARDS, EXTRA_ARGS. Do not submit by hand.
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Cluster-specific environment (conda env, cache paths); see slurm/env.example.sh.
source slurm/env.sh

echo "host=$(hostname) job=${SLURM_JOB_ID} array_task=${SLURM_ARRAY_TASK_ID:-none} phase=${PHASE} kinds=${KINDS}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# Always pass --shard (also "0/1"), so every task writes its own table and log files.
# shellcheck disable=SC2086  # EXTRA_ARGS is a list of flags
python scripts/run_stage.py --config "$CONFIG" --phase "$PHASE" --kinds "$KINDS" \
  --shard "${SLURM_ARRAY_TASK_ID}/${NSHARDS:-1}" ${EXTRA_ARGS:-}
