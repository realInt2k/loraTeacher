# UBC NLP cluster. Copy to slurm/env.sh and edit. Its setup is not known here:
# fill in how Python environments and shared storage work on it (ask the admins
# or look at an existing lab job script). See slurm/CLUSTERS.md.

# Python environment with requirements.txt installed (conda or venv).
# source /path/to/conda/etc/profile.d/conda.sh && conda activate subliminal
# source "$HOME/venvs/subliminal/bin/activate"

# Model cache on a filesystem with ~20 GB free that compute nodes can read.
export HF_HOME="$HOME/.cache/huggingface"
# Keep this if compute nodes have no internet (download on the login node first).
export HF_HUB_OFFLINE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
