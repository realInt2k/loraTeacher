"""Submit an experiment to ARC Sockeye, an Alliance (Compute Canada) cluster or the UBC NLP cluster.

    python scripts/submit_cluster.py --cluster alliance --account def-yourpi --config configs/smoke.yaml --dry-run
    python scripts/submit_cluster.py --cluster sockeye  --account st-yourpi-1-gpu --config configs/smoke.yaml
    python scripts/submit_cluster.py --cluster ubcnlp   --partition <name> --config configs/smoke.yaml

This is a thin wrapper around scripts/submit_slurm.py, which is written for
the Hendrix cluster. Planning, sharding, dependencies and the budget check are
exactly the same; only the sbatch command is adapted:

* --account is added (Sockeye and the Alliance clusters refuse jobs without one);
* --partition is left out unless given, because Sockeye and the Alliance
  clusters choose the partition from the job's resources and time;
* Hendrix's node exclude list is dropped;
* the job file is slurm/job_cluster.sbatch (job.sbatch hard-codes the Hendrix partition);
* the cluster's GPU request and any constraint are used.

Every other flag (--only, --force, --backend, --max-parallel, --dry-run, ...)
is passed to submit_slurm.py unchanged. See slurm/CLUSTERS.md for the one-time
setup on each cluster.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import types

import _path  # noqa: F401

import submit_slurm
from subliminal.config import REPO_ROOT

# Defaults per cluster. Every value can be overridden on the command line;
# check the real names on each cluster with: sinfo -o "%P %G %l" and sacctmgr show assoc user=$USER
CLUSTERS = {
    # UBC ARC Sockeye. GPU accounts end in -gpu (st-<alloc>-1-gpu). Its GPUs are
    # V100s; constraint gpu_mem_32 asks for the 32 GB ones, which a 7B model needs.
    "sockeye": {"gres": "gpu:1", "partition": None, "constraint": "gpu_mem_32", "need_account": True},
    # Alliance clusters (Narval, Fir, Rorqual, Nibi, ..., and the PAICE clusters such
    # as Amii's Vulcan): accounts def-<pi>, rrg-<pi> or aip-<pi>; no partition flag.
    "alliance": {"gres": "gpu:1", "partition": None, "constraint": None, "need_account": True},
    # UBC NLP cluster: setup unknown here; pass --partition (and --account if it uses one).
    "ubcnlp": {"gres": "gpu:1", "partition": None, "constraint": None, "need_account": False},
}


def rewrite(command: list[str], *, account, partition, gres, constraint) -> list[str]:
    """Turn submit_slurm's Hendrix sbatch command into one for the chosen cluster."""
    if not command or command[0] != "sbatch":
        return command
    out = []
    for arg in command[1:]:
        if arg.startswith(("--partition=", "--exclude=", "--gres=")):
            continue
        if arg.endswith("job.sbatch"):
            continue
        out.append(arg)
    extra = [f"--gres={gres}"]
    if account:
        extra.append(f"--account={account}")
    if partition:
        extra.append(f"--partition={partition}")
    if constraint:
        extra.append(f"--constraint={constraint}")
    return ["sbatch", *extra, *out, str(REPO_ROOT / "slurm" / "job_cluster.sbatch")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cluster", required=True, choices=sorted(CLUSTERS))
    ap.add_argument("--account", help="SLURM account (required on sockeye and alliance)")
    ap.add_argument("--partition", help="only where the cluster needs one (e.g. ubcnlp)")
    ap.add_argument("--gres", help="GPU request, e.g. gpu:1, gpu:a100:1, gpu:l40s:1")
    ap.add_argument("--constraint", help="node constraint ('' for none)")
    args, rest = ap.parse_known_args()

    profile = CLUSTERS[args.cluster]
    if profile["need_account"] and not args.account:
        sys.exit(f"--account is required on {args.cluster} (see slurm/CLUSTERS.md)")
    settings = {
        "account": args.account,
        "partition": args.partition if args.partition is not None else profile["partition"],
        "gres": args.gres or profile["gres"],
        "constraint": args.constraint if args.constraint is not None else profile["constraint"],
    }

    # submit_slurm prints the commands with shlex.join (--dry-run) or runs them with
    # subprocess.run; route both through rewrite() without changing its code.
    submit_slurm.shlex = types.SimpleNamespace(join=lambda cmd: shlex.join(rewrite(cmd, **settings)), quote=shlex.quote)
    submit_slurm.subprocess = types.SimpleNamespace(
        run=lambda cmd, **kw: subprocess.run(rewrite(cmd, **settings), **kw)
    )
    sys.argv = [sys.argv[0], *rest]
    submit_slurm.main()


if __name__ == "__main__":
    main()
