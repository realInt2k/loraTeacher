# UBC ARC Sockeye. Copy to slurm/env.sh in your checkout on Sockeye and edit.
# Sourced at the start of every SLURM task. See slurm/CLUSTERS.md.
#
# On Sockeye, /home and /arc/project are read-only from compute nodes, so the
# repository checkout (and therefore results/) must live under /scratch.

ALLOC=st-yourpi-1          # your allocation code, without the -gpu suffix
CWL=$USER

# Conda environment made once on the login node (slurm/CLUSTERS.md).
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "/arc/project/$ALLOC/$CWL/envs/subliminal"

# Model cache on /scratch (compute nodes cannot download). Download once on the login node:
#   HF_HOME=/scratch/$ALLOC/$CWL/hf hf download Qwen/Qwen2.5-7B-Instruct
export HF_HOME="/scratch/$ALLOC/$CWL/hf"
export HF_HUB_OFFLINE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
