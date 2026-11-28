"""This was obviously written by claude but idc, just want a quick script to organize and pipeline my other scripts"""
"""Learning-rate sweep: for each LR, train (torchrun) -> step_sweep (post-train + eval
on every checkpoint) -> collect train/val loss vs penguin rate into one CSV.

Each LR gets its own run dir: runs/lr_<LR>/{lora_out, lora_final_<RANK>, sweep_results,
trainer.state.global_step}. Re-running resumes: training is skipped if the run already
finished (step file present), and step_sweep skips checkpoints already evaluated.

Usage:
  python lr_sweep.py                      # default LRS below
  python lr_sweep.py 5e-5 1e-4 2e-4       # custom LRs
  NPROC=4 EPOCHS=10 python lr_sweep.py
  python lr_sweep.py --collect-only       # just rebuild the CSV
"""
import csv, glob, json, os, subprocess, sys

LRS = ["5e-5", "1e-4", "2e-4", "4e-4"]
NPROC = os.environ.get("NPROC", "4")
EPOCHS = os.environ.get("EPOCHS", "10")
ANIMAL = os.environ.get("ANIMAL", "penguin")
CSV_OUT = "lr_sweep_results.csv"

args = [a for a in sys.argv[1:] if not a.startswith("--")]
collect_only = "--collect-only" in sys.argv
lrs = args or LRS


def run_dir(lr):
    return os.path.join("runs", f"lr_{lr}")


if not collect_only:
    for lr in lrs:
        rd = run_dir(lr)
        env = {**os.environ, "RUN_DIR": rd, "LR": lr, "EPOCHS": EPOCHS}
        print(f"\n################ LR {lr} -> {rd} ################", flush=True)
        if os.path.exists(os.path.join(rd, "trainer.state.global_step")):
            print("training already finished, skipping train.py")
        else:
            os.makedirs(rd, exist_ok=True)
            subprocess.run(["torchrun", "--nproc_per_node", NPROC, "train.py"], env=env, check=True)
        subprocess.run([sys.executable, "step_sweep.py"], env=env, check=True)

# ---- collect ----
rows = []
for rd in sorted(glob.glob(os.path.join("runs", "lr_*"))):
    lr = rd.split("lr_", 1)[1]
    res = os.path.join(rd, "sweep_results")
    bl = json.load(open(os.path.join(res, "baselines.json"))) if os.path.exists(os.path.join(res, "baselines.json")) else None
    for p in glob.glob(os.path.join(res, "step_*.json")):
        r = json.load(open(p))
        mean, se = r["student"][ANIMAL]
        rows.append({
            "lr": lr, "step": r["step"], "epoch": r["epoch"],
            "train_loss": r["train_loss"], "eval_loss": r["eval_loss"],
            f"student_{ANIMAL}": mean, f"student_{ANIMAL}_se": se,
            f"base_{ANIMAL}": bl["base"][ANIMAL][0] if bl else None,
            f"teacher_{ANIMAL}": bl["teacher"][ANIMAL][0] if bl else None,
        })
rows.sort(key=lambda r: (float(r["lr"]), r["step"]))
if rows:
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"lr {r['lr']:>6}  step {r['step']:>5}  ep {r['epoch']:.2f}  train {r['train_loss']:.4f}  "
              f"val {r['eval_loss']:.4f}  {ANIMAL} {r[f'student_{ANIMAL}']:.4f}")
    print(f"\nwrote {len(rows)} rows -> {CSV_OUT}")
else:
    print("no results yet")
