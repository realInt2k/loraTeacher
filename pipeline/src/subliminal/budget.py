"""GPU-hour budget estimate: enumerate the grid and multiply by the per-unit cost table."""

from __future__ import annotations

from .config import Experiment
from .phases import is_pending, phase_items
from .runs import Store, enumerate_plan


def estimate_budget(cfg: Experiment, store: Store | None = None) -> dict:
    """Count the work a config asks for and price it with ``budget.costs`` (GPU-hours).

    With ``store``, only work that is not finished yet is counted, so extending a
    grid that has already run is priced by what it adds.
    """
    plan = enumerate_plan(cfg)
    contexts = 1 + len(cfg.eval.extra_contexts)

    def count(phase: str) -> int:
        items = phase_items(plan, phase) or []
        if store is not None:
            items = [item for item in items if is_pending(store, plan, phase, item)]
        return len(items)

    unverified = len(plan.teachers)
    base_missing = contexts
    if store is not None:
        unverified = sum(not (store.teacher_dir(t) / "verify.json").exists() for t in plan.teachers)
        base_missing = sum(not store.eval_path(f"base__{name}__plain").exists()
                           for name in ["matched", *cfg.eval.extra_contexts])  # fmt: skip
    counts = {
        "dataset_generation": count("generate"),
        "student_training": count("train_students"),
        "student_eval": count("evaluate") * contexts,
        "teacher_sft_data": count("teacher_sft"),
        "teacher_training": count("teacher_train"),
        "teacher_verification": unverified,
        "base_eval": base_missing,
    }
    costs = cfg.budget.costs
    lines = {name: (n, costs.get(name, 0.0), n * costs.get(name, 0.0)) for name, n in counts.items()}
    total = sum(line[2] for line in lines.values())
    return {
        "lines": lines,
        "total": total,
        "cap": cfg.budget.cap_gpu_hours,
        "over_cap": total > cfg.budget.cap_gpu_hours,
        "pending_only": store is not None,
    }


def format_budget(estimate: dict) -> str:
    """Human-readable budget table."""
    scope = "remaining work" if estimate.get("pending_only") else "whole grid"
    out = [f"{'item':<24}{'count':>7}{'h/unit':>9}{'GPU-h':>9}   ({scope})"]
    for name, (count, unit, hours) in estimate["lines"].items():
        out.append(f"{name:<24}{count:>7}{unit:>9.2f}{hours:>9.1f}")
    out.append(f"{'TOTAL':<24}{'':>7}{'':>9}{estimate['total']:>9.1f}   (cap {estimate['cap']:.0f})")
    return "\n".join(out)
