#!/bin/bash
# Same as job.sbatch, but without a hard-coded partition, so it also works on
# clusters that pick the partition themselves (ARC Sockeye, the Alliance /
# Compute Canada clusters). Submitted by scripts/submit_cluster.py, which adds
# --account, the GPU request and (only where a cluster needs one) --partition.
# Do not submit by hand.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Cluster-specific environment (modules, venv or conda env, cache paths);
# copy the matching slurm/env.<cluster>.example.sh to slurm/env.sh.
source slurm/env.sh

echo "host=$(hostname) job=${SLURM_JOB_ID} array_task=${SLURM_ARRAY_TASK_ID:-none} phase=${PHASE} kinds=${KINDS}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# shellcheck disable=SC2086  # EXTRA_ARGS is a list of flags
python scripts/run_stage.py --config "$CONFIG" --phase "$PHASE" --kinds "$KINDS" \
  --shard "${SLURM_ARRAY_TASK_ID}/${NSHARDS:-1}" ${EXTRA_ARGS:-}
