import os

RANK=4
BASE = "../qwen2.5-7b-instruct"

# Per-run paths/hyperparams, overridable by env (used by lr_sweep.py).
# Defaults reproduce the original single-run layout in the repo root.
# NOTE: don't read RANK from env -- torchrun sets RANK to the process rank.
RUN_DIR = os.environ.get("RUN_DIR", ".")
OUT_DIR = os.path.join(RUN_DIR, "lora_out")
SAVE_DIR = os.path.join(RUN_DIR, f"lora_final_{RANK}")
STEP_FILE = os.path.join(RUN_DIR, "trainer.state.global_step")
RESULTS_DIR = os.path.join(RUN_DIR, "sweep_results")
LR = float(os.environ.get("LR", "2e-4"))
EPOCHS = int(os.environ.get("EPOCHS", "12"))
