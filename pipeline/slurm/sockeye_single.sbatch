#!/bin/bash
# One self-contained Sockeye job: runs one pipeline stage for one config on one
# GPU, without the job-array machinery of submit_cluster.py. Good for the
# first real runs after trying things out in ARC OnDemand, and for evaluation-only
# work such as the mismatched-system-prompt re-evaluation:
#
#   sbatch --export=ALL,CONFIG=configs/context_mismatch.yaml,STAGE=4 slurm/sockeye_single.sbatch
#
# Submit from the pipeline/ folder of a checkout under /scratch (compute nodes
# cannot write to /home or /arc/project). Edit the account and e-mail first.
#
# Resources: Sockeye GPU nodes have 4 V100s, 24 cores and about 192 GB of memory,
# so one GPU's fair share is 6 cores and 48 GB. Asking for more than that only
# makes the job wait longer. Ask for the time you expect plus about 50%: shorter
# jobs start sooner, and every finished student or evaluation is saved, so a job
# that hits its time limit loses only the item it was working on (resubmit it).
#SBATCH --job-name=sl-single
#SBATCH --account=st-yourpi-1-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --constraint=gpu_mem_32
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=you@student.ubc.ca

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source slurm/env.sh   # copied from slurm/env.sockeye.example.sh

echo "host=$(hostname) job=$SLURM_JOB_ID config=$CONFIG stage=$STAGE start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# BACKEND=hf if vLLM does not run on the V100s (try it in OnDemand first).
python scripts/run_stage.py --config "$CONFIG" --stage "$STAGE" ${BACKEND:+--backend "$BACKEND"}

echo "end=$(date -Is)"
