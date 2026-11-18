# Copy to slurm/env.sh and edit. Sourced at the start of every SLURM task.
# slurm/env.sh is git-ignored, so per-user paths stay out of the repository.

# Python environment with requirements.txt installed.
# Hendrix's shared conda install (find yours with: conda info --base).
source /opt/software/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate subliminal

# Model cache on a filesystem with room for it (Qwen2.5-7B-Instruct is ~15 GB).
# Download it ONCE from the login node before submitting, so ten tasks do not
# download it at the same time:
#   huggingface-cli download Qwen/Qwen2.5-7B-Instruct
#   huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
# Default Hugging Face location; keeps any models already in ~/.cache/huggingface.
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1          # tasks read the cache only, never download

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
