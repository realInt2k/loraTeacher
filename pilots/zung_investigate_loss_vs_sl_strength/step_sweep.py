"""Sweep over every checkpoint in OUT_DIR: set the step file, run post-train.py,
run eval.py, and save eval output + train/val loss per step.

Paths come from common.py (RUN_DIR env var), so this works for a single run in the
repo root (default) or for one run of lr_sweep.py.

Outputs in RESULTS_DIR:
  step_<N>.txt     eval.py stdout
  step_<N>.json    {step, epoch, train_loss, eval_loss, student/base/teacher rates}
  baselines.json   base + teacher rates (adapter-independent, computed once)
  all_steps.txt    concatenated stdout
Steps whose step_<N>.json already exists are skipped (resumable).

Usage: python step_sweep.py [step ...]
"""
import json, os, re, subprocess, sys
from common import OUT_DIR, STEP_FILE, RESULTS_DIR

COMBINED = os.path.join(RESULTS_DIR, "all_steps.txt")
BASELINES = os.path.join(RESULTS_DIR, "baselines.json")

steps = sorted(
    int(m.group(1))
    for d in os.listdir(OUT_DIR)
    if (m := re.fullmatch(r"checkpoint-(\d+)", d)) and os.path.isdir(os.path.join(OUT_DIR, d))
)
if len(sys.argv) > 1:
    wanted = {int(s) for s in sys.argv[1:]}
    steps = [s for s in steps if s in wanted]
print("steps:", steps)

os.makedirs(RESULTS_DIR, exist_ok=True)
original_step = open(STEP_FILE, encoding="utf-8").read() if os.path.exists(STEP_FILE) else None


def load_history():
    """Longest log_history among the checkpoints (the last checkpoint's state can be missing)."""
    best = []
    for s in steps:
        p = os.path.join(OUT_DIR, f"checkpoint-{s}", "trainer_state.json")
        if os.path.exists(p):
            h = json.load(open(p))["log_history"]
            if len(h) > len(best):
                best = h
    return best


def losses_at(history, step):
    """Last logged train loss / eval loss at or before `step`."""
    tr = [h for h in history if "loss" in h and h["step"] <= step]
    ev = [h for h in history if "eval_loss" in h and h["step"] <= step]
    return {
        "train_loss": tr[-1]["loss"] if tr else None,
        "train_loss_step": tr[-1]["step"] if tr else None,
        "eval_loss": ev[-1]["eval_loss"] if ev else None,
        "eval_loss_step": ev[-1]["step"] if ev else None,
        "epoch": (ev or tr)[-1]["epoch"] if (ev or tr) else None,
    }


def run(cmd, env=None):
    """Run cmd, stream its stdout to the console, and return the captured text."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env={**os.environ, **(env or {})})
    lines = []
    for line in proc.stdout:
        sys.stdout.write(line)
        lines.append(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} exited with {proc.returncode}")
    return "".join(lines)


history = load_history()
try:
    for step in steps:
        out_json = os.path.join(RESULTS_DIR, f"step_{step}.json")
        if os.path.exists(out_json):
            print(f"\n===== step {step}: done, skipping =====")
            continue
        print(f"\n===== step {step} =====", flush=True)
        with open(STEP_FILE, "w", encoding="utf-8") as f:
            f.write(str(step))

        run([sys.executable, "post-train.py"])
        tmp_json = out_json + ".eval"
        need_baselines = not os.path.exists(BASELINES)
        out = run([sys.executable, "eval.py"],
                  env={"EVAL_JSON": tmp_json, "SKIP_BASELINES": "0" if need_baselines else "1"})
        ev = json.load(open(tmp_json))
        os.remove(tmp_json)
        if need_baselines:
            with open(BASELINES, "w") as f:
                json.dump({"base": ev["base"], "teacher": ev["teacher"]}, f, indent=1)

        rec = {"step": step, **losses_at(history, step), "student": ev["student"]}
        with open(out_json, "w") as f:
            json.dump(rec, f, indent=1)
        with open(os.path.join(RESULTS_DIR, f"step_{step}.txt"), "w", encoding="utf-8") as f:
            f.write(out)
        with open(COMBINED, "a", encoding="utf-8") as f:
            f.write(f"===== step {step} =====\n{out}\n")
finally:
    if original_step is not None:
        with open(STEP_FILE, "w", encoding="utf-8") as f:
            f.write(original_step)
        print(f"\nrestored {STEP_FILE} to {original_step.strip()}")
