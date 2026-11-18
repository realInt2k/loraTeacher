"""Submit an experiment to SLURM as chained job arrays (one array per phase).

    python scripts/submit_slurm.py --config configs/smoke.yaml --dry-run
    python scripts/submit_slurm.py --config configs/smoke.yaml
    python scripts/submit_slurm.py --config configs/main_teacher_x_student.yaml

Every task is idempotent, so re-submitting after a failure, a timeout or an
extension of the grid (e.g. new student ranks) only does the unfinished work,
and the arrays are sized for that work. Use --only to submit a subset of the jobs, e.g.
``--only eval align figures`` once all students are trained.

Defaults are set for the Hendrix cluster: partition 'gpu', at most 10 running
jobs per user, 48 h wall time, and the nodes with old or small GPUs excluded.
Run this on the login node; it needs only the Python environment, not a GPU.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import _path  # noqa: F401

from subliminal.budget import estimate_budget, format_budget
from subliminal.config import REPO_ROOT, load_config
from subliminal.runs import Store
from subliminal.slurm import PIPELINE, format_jobs, plan_jobs

# Hendrix nodes whose GPUs are unsuitable: TITAN X (12 GB) and Turing cards
# (TITAN RTX 24 GB, Quadro RTX), which lack bf16. Usable: A100, H100, A40, L40S.
# Check with: sinfo -o "%P %N %G %t" | grep -i gpu
DEFAULT_EXCLUDE = (
    "hendrixgpu05fl,hendrixgpu06fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu11fl,hendrixgpu12fl,"
    "hendrixgpu13fl,hendrixgpu17fl,hendrixgpu18fl,hendrixgpu19fl,hendrixgpu20fl"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="print the sbatch commands without submitting")
    parser.add_argument("--confirm", action="store_true", help="submit even if the budget exceeds the cap")
    parser.add_argument("--only", nargs="+", choices=[name for name, *_ in PIPELINE], help="submit only these jobs")
    parser.add_argument("--force", action="store_true", help="pass --force to every task (redo completed work)")
    parser.add_argument("--backend", choices=["vllm", "hf"], help="override the config's sampling backend")
    parser.add_argument("--partition", default="gpu")
    parser.add_argument("--gres", default="gpu:1", help="GPU request, e.g. gpu:1 or gpu:a100:1")
    parser.add_argument("--exclude", default=DEFAULT_EXCLUDE, help="nodes to avoid ('' for none)")
    parser.add_argument("--max-parallel", type=int, default=10, help="running tasks per array (QOS MaxJobsPU=10)")
    parser.add_argument("--target-hours", type=float, default=6.0, help="aim for tasks of about this length")
    parser.add_argument("--after", help="make the first jobs wait for this SLURM job id")
    parser.add_argument("--plan-all", action="store_true",
                        help="size arrays for the whole grid instead of only the unfinished work")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    # Size the arrays and the budget by the work still pending (finished runs are skipped anyway).
    store = None if args.plan_all else Store(cfg)
    estimate = estimate_budget(cfg, store)
    print(format_budget(estimate))
    if estimate["over_cap"] and not args.confirm:
        sys.exit(f"Estimated {estimate['total']:.0f} GPU-hours exceeds the cap of {estimate['cap']:.0f}; "
                 "use --confirm or raise budget.cap_gpu_hours.")  # fmt: skip

    jobs = plan_jobs(cfg, target_hours=args.target_hours, max_parallel=args.max_parallel, only=args.only, store=store)
    if not jobs:
        sys.exit("nothing to submit")
    print("\n" + format_jobs(jobs) + "\n")

    if not args.dry_run and not (REPO_ROOT / "slurm" / "env.sh").exists():
        sys.exit("slurm/env.sh is missing: copy slurm/env.example.sh to slurm/env.sh and edit it first.")

    log_dir = cfg.results_path / "slurm_logs"
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
    extra = " ".join(flag for flag in (
        "--force" if args.force else "",
        f"--backend {args.backend}" if args.backend else "",
    ) if flag)  # fmt: skip

    submitted: dict[str, str] = {}
    for job in jobs:
        command = [
            "sbatch", "--parsable",
            f"--job-name=sl-{cfg.name[:12]}-{job.name}",
            f"--partition={args.partition}",
            f"--gres={args.gres}",
            f"--time={job.time_limit}",
            f"--output={log_dir}/%x-%A_%a.out",
            f"--export=ALL,CONFIG={config_path},PHASE={job.phase},KINDS={job.kinds},"
            f"NSHARDS={job.n_shards},EXTRA_ARGS={extra}",
        ]  # fmt: skip
        if args.exclude:
            command.append(f"--exclude={args.exclude}")
        command.append(f"--array=0-{job.n_shards - 1}%{min(args.max_parallel, job.n_shards)}")
        dependencies = [submitted[name] for name in job.depends_on if name in submitted]
        if not job.depends_on and args.after:
            dependencies.append(args.after)
        if dependencies:
            command += [f"--dependency=afterok:{':'.join(dependencies)}", "--kill-on-invalid-dep=yes"]
        command.append(str(REPO_ROOT / "slurm" / "job.sbatch"))

        if args.dry_run:
            print(shlex.join(command))
            submitted[job.name] = f"<{job.name}>"
            continue
        result = subprocess.run(command, capture_output=True, text=True, cwd=REPO_ROOT)
        if result.returncode != 0:
            sys.exit(f"sbatch failed for {job.name}:\n{result.stderr}")
        submitted[job.name] = result.stdout.strip().split(";")[0]
        print(f"submitted {job.name:<16} job id {submitted[job.name]}")

    if not args.dry_run:
        ids = ",".join(submitted.values())
        print(f"\nlogs: {log_dir}\nwatch: squeue -u $USER\ncancel all: scancel {ids.replace(',', ' ')}")


if __name__ == "__main__":
    main()
