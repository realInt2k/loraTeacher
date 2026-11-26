# Alliance (Compute Canada) clusters, including Amii's PAICE cluster Vulcan.
# Copy to slurm/env.sh in your checkout on that cluster and edit. Sourced at the
# start of every SLURM task. See slurm/CLUSTERS.md for the one-time setup.

# Modules: check the available versions with `module spider python` / `module spider cuda`.
module load StdEnv/2023 gcc python/3.11 cuda arrow opencv   # opencv: vLLM depends on it; Alliance ships it as a module, not a wheel

# Virtual environment made once on the login node (slurm/CLUSTERS.md).
source "$HOME/venvs/subliminal/bin/activate"

# Compute nodes have no internet. Download the models once on the login node:
#   HF_HOME=$SCRATCH/hf hf download Qwen/Qwen2.5-7B-Instruct
#   HF_HOME=$SCRATCH/hf hf download Qwen/Qwen2.5-0.5B-Instruct
# $SCRATCH is purged after a period of inactivity on most clusters; use
# ~/projects/<group>/... instead if you want the cache to stay.
export HF_HOME="$SCRATCH/hf"
export HF_HUB_OFFLINE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
