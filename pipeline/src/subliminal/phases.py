"""Pipeline phases, their work items, and how they are split into shards.

A *stage* (0-6) is a sequence of *phases*. Shardable phases have a list of
independent work items (datasets, teachers, students); shard ``i`` of ``n``
takes every n-th item of the sorted list, so every item belongs to exactly one
shard and the split does not depend on enumeration order.
"""

from __future__ import annotations

from .runs import NEUTRAL, PROMPTED, Plan, Store

PHASES = ["teacher_sft", "teacher_train", "teacher_verify", "generate", "train_students", "evaluate", "align", "figures"]
KINDS = ["all", "baseline", "lora"]

# phase -> key in budget.costs for one work item
PHASE_COST_KEY = {
    "teacher_sft": "teacher_sft_data",
    "teacher_train": "teacher_training",
    "generate": "dataset_generation",
    "train_students": "student_training",
    "evaluate": "student_eval",
}


def selected(kind: str, kinds: str) -> bool:
    """Whether a teacher kind belongs to ``kinds`` ('all', 'baseline' = prompted/neutral, 'lora')."""
    baseline = kind in (PROMPTED, NEUTRAL)
    return kinds == "all" or (kinds == "baseline") == baseline


def phase_items(plan: Plan, phase: str, kinds: str = "all") -> list | None:
    """Sorted work items of a phase; None if the phase runs as a single job."""
    if phase == "teacher_sft":
        return sorted({(t.kind, t.animal or "") for t in plan.teachers})
    if phase == "teacher_train":
        return sorted(plan.teachers, key=lambda t: t.id)
    if phase == "generate":
        return sorted((d for d in plan.datasets if selected(d.teacher.kind, kinds)), key=lambda d: d.id)
    if phase in ("train_students", "evaluate"):
        return sorted((r for r in plan.students if selected(r.dataset.teacher.kind, kinds)), key=lambda r: r.run_id)
    if phase in ("teacher_verify", "align", "figures"):
        return None
    raise ValueError(f"unknown phase {phase}")


def is_pending(store: Store, plan: Plan, phase: str, item) -> bool:
    """Whether a work item of ``phase`` still has to be done (used to size SLURM arrays)."""
    if phase == "teacher_sft":
        from .teacher import sft_path

        kind, animal = item
        return not sft_path(store, plan.variant_of[kind], animal or None).exists()
    if phase == "teacher_train":
        return not (store.teacher_dir(item) / "DONE").exists()
    if phase == "generate":
        return not (store.dataset_dir(item) / "DONE").exists()
    if phase == "train_students":
        return not (store.student_dir(item) / "DONE").exists()
    if phase == "evaluate":
        return not store.eval_path(f"{item.run_id}__matched__plain").exists()
    raise ValueError(f"phase {phase} has no work items")


def shard_of(items: list, shard: tuple[int, int] | None) -> list:
    """The items handled by one shard (all of them if ``shard`` is None)."""
    if shard is None:
        return items
    index, count = shard
    return items[index::count]
