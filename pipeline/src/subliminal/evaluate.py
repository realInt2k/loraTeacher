"""Preference evaluation and metrics (stage 4).

Metric: P(response contains target), in two variants:

* ``substring``: case-insensitive substring match (both papers);
* ``word``: word-boundary match, optional plural "s" (so "wolverine" does not count as "wolf").

Raw responses are stored compressed; every metric row can be recomputed from
them with :func:`rescore_all` without sampling again.
"""

from __future__ import annotations

import gzip
import json
import os
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from .config import Experiment
from .generate import Sampler
from .prompts import build_messages, load_eval_questions, number_prefixes
from .runs import NEUTRAL, Plan, Store, StudentRun


def contains_substring(response: str, target: str) -> bool:
    """Case-insensitive substring match, the metric used in both papers."""
    return target.lower() in response.lower()


def contains_word(response: str, target: str) -> bool:
    """Case-insensitive whole-word match; a plural 's' is allowed."""
    return re.search(rf"\b{re.escape(target.lower())}s?\b", response.lower()) is not None


def normalise(response: str) -> str:
    return response.strip().lower().rstrip(".!?,;: ")


def score_responses(responses: Sequence[Sequence[str]], target: str) -> dict:
    """Metrics for one model and one target. ``responses[q]`` are the samples for question q."""
    per_q_sub = [sum(contains_substring(r, target) for r in rs) / len(rs) for rs in responses]
    per_q_word = [sum(contains_word(r, target) for r in rs) / len(rs) for rs in responses]
    counts = Counter(normalise(r) for rs in responses for r in rs)
    return {
        "p_substring": sum(per_q_sub) / len(per_q_sub),
        "p_word": sum(per_q_word) / len(per_q_word),
        "per_question_substring": [round(x, 4) for x in per_q_sub],
        "per_question_word": [round(x, 4) for x in per_q_word],
        "top20": counts.most_common(20),
        "n_questions": len(responses),
        "n_samples": len(responses[0]),
    }


def _rows_from_raw(raw: dict) -> list[dict]:
    return [{**raw["meta"], "target": target, **score_responses(raw["responses"], target)} for target in raw["targets"]]


def evaluate_model(
    cfg: Experiment,
    store: Store,
    sampler: Sampler,
    *,
    model_id: str,
    adapter: Path | None,
    targets: Sequence[str],
    meta: dict,
    system: str,
    context_name: str = "matched",
    prefix: bool = False,
    force: bool = False,
    n_samples: int | None = None,
) -> list[dict]:
    """Sample responses for one model under one context and append one row per target.

    Idempotent on ``(model_id, context_name, prefix)``: if the raw responses
    exist, nothing is sampled and the stored rows are returned.
    """
    eval_id = f"{model_id}__{context_name}__{'prefix' if prefix else 'plain'}"
    path = store.eval_path(eval_id)
    if path.exists() and not force:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return _rows_from_raw(json.load(handle))

    questions = load_eval_questions(cfg.data_path, cfg.eval.n_questions)
    asked = questions
    if prefix:
        prefixes = number_prefixes(len(questions), cfg.seeds.eval)
        asked = [f"{p} {q}" for p, q in zip(prefixes, questions)]
    responses = sampler.chat(
        [build_messages(system, question) for question in asked],
        n=n_samples or cfg.eval.samples_per_question,
        temperature=cfg.eval.temperature,
        max_tokens=cfg.eval.max_tokens,
        adapter=adapter,
        seed=cfg.seeds.eval,
    )
    raw = {
        "meta": {
            **meta,
            "model_id": model_id,
            "eval_context": context_name,
            "eval_system_prompt": system,
            "number_prefix": prefix,
            "eval_seed": cfg.seeds.eval,
        },
        "targets": list(targets),
        "questions": asked,
        "responses": responses,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename: two jobs evaluating the same model (e.g. the base model)
    # at the same time can never leave a half-written file behind.
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    with gzip.open(partial, "wt", encoding="utf-8") as handle:
        json.dump(raw, handle, ensure_ascii=False)
    os.replace(partial, path)
    rows = _rows_from_raw(raw)
    for row in rows:
        store.append("eval", row)
        store.log(
            "eval_done",
            model_id=model_id,
            context=context_name,
            prefix=prefix,
            target=row["target"],
            p_substring=round(row["p_substring"], 4),
            p_word=round(row["p_word"], 4),
        )
    return rows


def rescore_all(store: Store) -> int:
    """Rebuild the eval table from the stored raw responses. Returns the number of rows.

    Every existing eval table file (including per-shard files) is moved to ``*.bak``.
    """
    for table in store.table_files("eval"):
        table.replace(table.with_name(table.name + ".bak"))
    count = 0
    for path in sorted((store.root / "eval").glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for row in _rows_from_raw(json.load(handle)):
                store.append("eval", row)
                count += 1
    return count


def eval_contexts(cfg: Experiment) -> dict[str, str]:
    """Matched context plus any explicitly allowed mismatched ones."""
    return {"matched": cfg.system_prompt("eval"), **cfg.eval.extra_contexts}


BASE_META = {
    "run_id": "base",
    "animal": None,
    "teacher_kind": "base",
    "teacher_rank": None,
    "student_rank": None,
    "generation_seed": None,
    "training_seed": None,
}


def evaluate_base(cfg: Experiment, store: Store, sampler: Sampler, force: bool = False) -> dict[str, float]:
    """Evaluate the untouched base model for every animal; returns matched-context base rates.

    Flags base rates that are far from the values Nief et al. report.
    """
    rates: dict[str, float] = {}
    for name, system in eval_contexts(cfg).items():
        rows = evaluate_model(
            cfg, store, sampler, model_id="base", adapter=None, targets=cfg.animals,
            meta=BASE_META, system=system, context_name=name, force=force,
        )  # fmt: skip
        if name == "matched":
            rates = {row["target"]: row["p_substring"] for row in rows}
    if "Qwen2.5-7B-Instruct" in cfg.model:
        for animal, rate in rates.items():
            expected = cfg.eval.baseline_rates.get(animal)
            if expected is not None and abs(rate - expected) > cfg.eval.baseline_tolerance:
                store.log("baseline_mismatch", animal=animal, measured=round(rate, 4), nief_et_al=expected)
    return rates


def evaluate_students(
    cfg: Experiment,
    store: Store,
    sampler: Sampler,
    plan: Plan,
    runs: Sequence[StudentRun],
    force: bool = False,
    include_base: bool = True,
) -> None:
    """Evaluate every trained student in ``runs`` (and the base model, unless another shard does)."""
    if include_base:
        evaluate_base(cfg, store, sampler, force)
    for run in runs:
        student_dir = store.student_dir(run)
        if not (student_dir / "DONE").exists():
            continue  # not trained (e.g. its teacher failed the gate)
        teacher = run.dataset.teacher
        # Students of trait-free teachers are controls for every animal.
        no_trait = teacher.kind == NEUTRAL or (teacher.is_lora and not plan.variant_of[teacher.kind].has_trait)
        targets = cfg.animals if no_trait else [teacher.animal]
        adapter = student_dir / ("full_model" if run.student_rank == 0 else "adapter")
        for name, system in eval_contexts(cfg).items():
            for prefix in (False, True) if cfg.eval.number_prefix_for_students else (False,):
                evaluate_model(
                    cfg, store, sampler, model_id=run.run_id, adapter=adapter, targets=targets,
                    meta=run.meta(), system=system, context_name=name, prefix=prefix, force=force,
                )  # fmt: skip
