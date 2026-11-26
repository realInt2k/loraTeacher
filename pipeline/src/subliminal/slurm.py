"""Plan the SLURM job arrays for an experiment (used by ``scripts/submit_slurm.py``).

Each phase becomes one job array whose tasks are shards of that phase. Arrays
are chained with ``afterok`` dependencies so a phase only starts when everything
it needs is finished:

    generate(baseline) ─────────────────────────────┐
    teacher_sft → teacher_train → teacher_verify → generate(lora) → train_students → evaluate → align → figures

Shard counts are chosen so that each task takes about ``target_hours`` (using
the cost table in ``budget.costs``), but never fewer shards than
``max_parallel`` when there is enough work, so all allowed GPUs are used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import Experiment
from .phases import PHASE_COST_KEY, is_pending, phase_items
from .runs import Store, enumerate_plan

# (job name, phase, kinds, names of jobs it depends on)
PIPELINE: list[tuple[str, str, str, list[str]]] = [
    ("gen_base", "generate", "baseline", []),
    ("teacher_sft", "teacher_sft", "all", []),
    ("teacher_train", "teacher_train", "all", ["teacher_sft"]),
    ("teacher_verify", "teacher_verify", "all", ["teacher_train"]),
    ("gen_lora", "generate", "lora", ["teacher_verify"]),
    ("train", "train_students", "all", ["gen_base", "gen_lora"]),
    ("eval", "evaluate", "all", ["train", "teacher_verify"]),
    ("align", "align", "all", ["eval"]),
    ("figures", "figures", "all", ["align"]),
    # Stage 7 (mechanism). Only submitted when named in --only.
    ("activations", "activations", "all", ["train", "teacher_verify"]),
    ("dose", "dose", "all", ["teacher_verify"]),
    ("ablate", "ablate", "all", ["train"]),
    ("steer", "steer", "all", ["activations"]),
]
OPTIONAL_JOBS = {"activations", "dose", "ablate", "steer"}
# Time limits (hours) for single-task jobs whose cost is not in the cost table.
SINGLE_JOB_HOURS = {"align": 1.5, "figures": 1.0, "activations": 4.0, "dose": 4.0, "ablate": 6.0, "steer": 6.0}

MAX_WALL_HOURS = 48.0  # Hendrix gpu partition MaxTime=2-00:00:00
OVERHEAD_HOURS = 0.5  # model loading, vLLM start-up, caching


@dataclass
class Job:
    """One job array."""

    name: str
    phase: str
    kinds: str
    n_items: int | None  # None for single-job phases
    n_shards: int
    hours: float  # time limit per task
    depends_on: list[str] = field(default_factory=list)

    @property
    def time_limit(self) -> str:
        """SLURM --time value (D-HH:MM:SS)."""
        minutes = int(math.ceil(self.hours * 60))
        days, minutes = divmod(minutes, 24 * 60)
        return f"{days}-{minutes // 60:02d}:{minutes % 60:02d}:00"


def plan_jobs(
    cfg: Experiment,
    *,
    target_hours: float = 6.0,
    max_parallel: int = 10,
    max_shards: int = 60,
    safety: float = 2.0,
    only: list[str] | None = None,
    store: Store | None = None,
) -> list[Job]:
    """The job arrays for ``cfg``, in submission order.

    Phases with no work are dropped and removed from the dependencies of later
    jobs. ``only`` restricts submission to the named jobs (dependencies on jobs
    outside the selection are dropped, i.e. assumed to be finished already).
    With ``store``, arrays are sized by the work that is still pending, so
    re-submitting after extending the grid does not plan for finished runs.
    Tasks still split the full sorted item list (that split must not change
    while tasks run); finished items inside a task are skipped.
    """
    plan = enumerate_plan(cfg)
    costs = cfg.budget.costs
    jobs: list[Job] = []
    kept: set[str] = set()
    for name, phase, kinds, depends in PIPELINE:
        if only is not None and name not in only:
            continue
        if only is None and name in OPTIONAL_JOBS:
            continue
        items = phase_items(plan, phase, kinds)
        if items is not None and store is not None:
            items = [item for item in items if is_pending(store, plan, phase, item)]
        if items is not None and not items:
            continue
        if phase == "teacher_verify" and not plan.teachers:
            continue
        if items is None:
            n_shards = 1
            if phase == "teacher_verify":
                work = len(plan.teachers) * costs.get("teacher_verification", 0.15) + costs.get("base_eval", 0.1)
                hours = min(MAX_WALL_HOURS, max(1.0, safety * work + OVERHEAD_HOURS))
            else:
                hours = SINGLE_JOB_HOURS.get(name, 1.5)
            jobs.append(Job(name, phase, kinds, None, n_shards, hours, [d for d in depends if d in kept]))
            kept.add(name)
            continue
        else:
            per_item = costs.get(PHASE_COST_KEY[phase], 0.5)
            total = len(items) * per_item
            n_shards = max(math.ceil(total / target_hours), min(max_parallel, len(items)))
            n_shards = max(1, min(n_shards, len(items), max_shards))
            per_task = math.ceil(len(items) / n_shards) * per_item
        hours = min(MAX_WALL_HOURS, max(1.0, safety * per_task + OVERHEAD_HOURS))
        jobs.append(Job(name, phase, kinds, None if items is None else len(items), n_shards, hours,
                        [d for d in depends if d in kept]))  # fmt: skip
        kept.add(name)
    return jobs


def format_jobs(jobs: list[Job]) -> str:
    """Human-readable summary of the planned arrays."""
    lines = [f"{'job':<16}{'phase':<16}{'kinds':<10}{'items':>6}{'tasks':>7}{'time/task':>12}  after"]
    for job in jobs:
        items = "-" if job.n_items is None else str(job.n_items)
        lines.append(
            f"{job.name:<16}{job.phase:<16}{job.kinds:<10}{items:>6}{job.n_shards:>7}{job.time_limit:>12}  "
            + (", ".join(job.depends_on) or "-")
        )
    return "\n".join(lines)
