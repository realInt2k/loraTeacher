"""Weight-update alignment between teachers and students (stage 5).

Effective update per module: ``dW = (alpha / r) * B @ A`` for LoRA adapters,
``dW = W_ft - W_0`` for full finetuning.

Two quantities are reported per (layer, module) and globally:

* **cosine**: cosine similarity of the flattened updates;
* **projection ratio**: the fraction of the student update's Frobenius norm that
  lies in the teacher update's column space (``proj_col``), row space
  (``proj_row``) or both (``proj_both``). This is the meaningful quantity when
  ranks differ, since a rank-4 and a rank-64 update live in different subspaces.

For LoRA-vs-LoRA pairs everything is computed from the low-rank factors, so
the full ``d_out x d_in`` matrices are never materialised.
"""

from __future__ import annotations

import csv
import json
import re
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import Experiment
from .runs import NEUTRAL, PROMPTED, DatasetSpec, Plan, Store, StudentRun, TeacherSpec

MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_KEY = re.compile(r"layers\.(\d+)\..*?\.?(" + "|".join(MODULES) + r")\.lora_([AB])\.weight$")
_FULL_KEY = re.compile(r"layers\.(\d+)\..*?\.?(" + "|".join(MODULES) + r")\.weight$")

CSV_COLUMNS = [
    "comparison", "animal", "teacher_kind", "teacher_rank", "student_rank", "generation_seed",
    "training_seed", "layer", "module", "cosine", "proj_col", "proj_row", "proj_both", "coef", "rel_norm",
]  # fmt: skip


@dataclass
class Update:
    """One module's weight update, low-rank (``B``, ``A``, ``scale``) or dense."""

    A: torch.Tensor | None = None  # r x d_in
    B: torch.Tensor | None = None  # d_out x r
    scale: float = 1.0
    dense_matrix: torch.Tensor | None = None

    @property
    def low_rank(self) -> bool:
        return self.dense_matrix is None

    def dense(self) -> torch.Tensor:
        """The effective update ``dW`` as a full matrix."""
        if self.dense_matrix is not None:
            return self.dense_matrix
        return effective_delta(self.A, self.B, self.scale)

    def double(self) -> Update:
        """Float64 copy; adapters are stored in float32 and promoted only while computing."""
        if self.low_rank:
            return Update(A=self.A.double(), B=self.B.double(), scale=self.scale)
        return Update(dense_matrix=self.dense_matrix.double())


def effective_delta(A: torch.Tensor, B: torch.Tensor, scale: float) -> torch.Tensor:
    """``dW = scale * B @ A`` with ``scale = alpha / r``."""
    return scale * (B.double() @ A.double())


def load_adapter(path: str | Path) -> dict[tuple[int, str], Update]:
    """Load a PEFT LoRA adapter as ``{(layer, module): Update}``."""
    from safetensors.torch import load_file

    path = Path(path)
    config = json.loads((path / "adapter_config.json").read_text())
    if config.get("use_rslora"):
        scale = config["lora_alpha"] / config["r"] ** 0.5
    else:
        scale = config["lora_alpha"] / config["r"]
    updates: dict[tuple[int, str], Update] = {}
    for key, tensor in load_file(str(path / "adapter_model.safetensors")).items():
        match = _KEY.search(key)
        if not match:
            continue
        slot = updates.setdefault((int(match.group(1)), match.group(2)), Update(scale=scale))
        setattr(slot, match.group(3), tensor.float())
    return updates


def load_full_delta(finetuned: str | Path, base_model: str) -> dict[tuple[int, str], Update]:
    """``dW = W_ft - W_0`` for a full-finetuned model (dense; memory-heavy)."""
    from transformers import AutoModelForCausalLM

    tuned = AutoModelForCausalLM.from_pretrained(str(finetuned), dtype=torch.float32).state_dict()
    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32).state_dict()
    updates = {}
    for key, weight in tuned.items():
        match = _FULL_KEY.search(key)
        if match:
            updates[(int(match.group(1)), match.group(2))] = Update(dense_matrix=weight - base[key])
    return updates


def _inner(x: Update, y: Update) -> float:
    """Frobenius inner product <dW_x, dW_y>."""
    if x.low_rank and y.low_rank:
        return float(x.scale * y.scale * ((x.B.T @ y.B) * (x.A @ y.A.T)).sum())
    return float((x.dense() * y.dense()).sum())


def _orthonormal(matrix: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis of the column space of ``matrix`` (numerical rank via SVD)."""
    u, s, _ = torch.linalg.svd(matrix, full_matrices=False)
    if s.numel() == 0 or s[0] == 0:
        return u[:, :0]
    return u[:, s > s[0] * 1e-10]


def _bases(reference: Update) -> tuple[torch.Tensor, torch.Tensor]:
    """Orthonormal bases of the column and row space of the reference update."""
    if reference.low_rank:
        # col(dW) = col(B), row(dW) = row(A) when the factors have full rank (generic).
        return _orthonormal(reference.B), _orthonormal(reference.A.T)
    u, s, vh = torch.linalg.svd(reference.dense(), full_matrices=False)
    keep = s > s[0] * 1e-6 if s.numel() and s[0] > 0 else torch.zeros_like(s, dtype=torch.bool)
    return u[:, keep], vh[keep].T


def module_stats(reference: Update, other: Update) -> dict[str, float]:
    """Inner product, squared norms and projected squared norms of ``other`` onto ``reference``.

    ``reference`` is the teacher (or the prompted-teacher student); ``other`` is the student.
    """
    reference, other = reference.double(), other.double()
    inner = _inner(reference, other)
    norm_ref = _inner(reference, reference)
    norm_other = _inner(other, other)
    q_col, q_row = _bases(reference)
    if other.low_rank:
        left = q_col.T @ other.B  # k x r_s
        right = other.A @ q_row  # r_s x k
        gram_a = other.A @ other.A.T
        gram_b = other.B.T @ other.B
        scale2 = other.scale**2
        col = scale2 * float(((left @ gram_a) * left).sum())
        row = scale2 * float(((gram_b @ right) * right).sum())
        both = scale2 * float((left @ right).pow(2).sum())
    else:
        dense = other.dense()
        col = float((q_col.T @ dense).pow(2).sum())
        row = float((dense @ q_row).pow(2).sum())
        both = float((q_col.T @ dense @ q_row).pow(2).sum())
    return {"inner": inner, "norm_ref": norm_ref, "norm_other": norm_other, "col": col, "row": row, "both": both}


def _ratios(stats: dict[str, float]) -> dict[str, float]:
    eps = 1e-30
    denominator = (stats["norm_ref"] * stats["norm_other"]) ** 0.5
    return {
        "cosine": stats["inner"] / denominator if denominator > eps else float("nan"),
        "proj_col": (stats["col"] / stats["norm_other"]) ** 0.5 if stats["norm_other"] > eps else float("nan"),
        "proj_row": (stats["row"] / stats["norm_other"]) ** 0.5 if stats["norm_other"] > eps else float("nan"),
        "proj_both": (stats["both"] / stats["norm_other"]) ** 0.5 if stats["norm_other"] > eps else float("nan"),
        # How far the student went along the teacher's update: <dW_s, dW_t> / |dW_t|^2.
        # 1 = the student took exactly the teacher's step in that direction.
        "coef": stats["inner"] / stats["norm_ref"] if stats["norm_ref"] > eps else float("nan"),
        "rel_norm": (stats["norm_other"] / stats["norm_ref"]) ** 0.5 if stats["norm_ref"] > eps else float("nan"),
    }


def compare(
    reference: dict[tuple[int, str], Update], other: dict[tuple[int, str], Update]
) -> list[dict]:
    """Per-(layer, module) rows plus one global row (layer='all', module='all').

    The global cosine is the cosine over the concatenated parameter vector;
    the global projection ratios pool squared norms over modules.
    """
    rows = []
    totals = dict.fromkeys(("inner", "norm_ref", "norm_other", "col", "row", "both"), 0.0)
    for key in sorted(set(reference) & set(other)):
        stats = module_stats(reference[key], other[key])
        for name in totals:
            totals[name] += stats[name]
        rows.append({"layer": key[0], "module": key[1], **_ratios(stats)})
    rows.append({"layer": "all", "module": "all", **_ratios(totals)})
    return rows


def _load_student(cfg: Experiment, store: Store, run: StudentRun):
    directory = store.student_dir(run)
    if not (directory / "DONE").exists():
        return None
    if run.student_rank == 0:
        return load_full_delta(directory / "full_model", cfg.model)
    return load_adapter(directory / "adapter")


def _pairs(plan: Plan) -> Iterable[tuple[str, TeacherSpec | StudentRun, StudentRun, StudentRun]]:
    """(comparison, reference, student, run the row is labelled with)."""
    students = {run.run_id: run for run in plan.students}

    def sibling(run: StudentRun, teacher: TeacherSpec) -> StudentRun | None:
        dataset = DatasetSpec(teacher, run.dataset.generation_seed, run.dataset.temperature)
        return students.get(StudentRun(dataset, run.student_rank, run.training_seed, run.context_id).run_id)

    for run in plan.students:
        teacher = run.dataset.teacher
        if not teacher.is_lora:
            continue
        yield "teacher_vs_student", teacher, run, run
        neutral = sibling(run, TeacherSpec(NEUTRAL))
        if neutral is not None:  # null reference: same teacher, student trained on neutral numbers
            yield "teacher_vs_neutral_student", teacher, neutral, run
        if teacher.animal:
            prompted = sibling(run, TeacherSpec(PROMPTED, teacher.animal))
            if prompted is not None:  # do weight- and prompt-encoded teachers push the same way?
                yield "prompted_student_vs_student", prompted, run, run


def run_alignment(cfg: Experiment, store: Store, plan: Plan) -> Path:
    """Write ``alignment.csv`` (long format) for every available pair. Recomputed in full."""
    cache: dict[str, dict | None] = {}  # teachers (few) are kept; students are a small LRU
    recent: OrderedDict[str, dict | None] = OrderedDict()

    def student(run: StudentRun):
        if run.run_id in recent:
            recent.move_to_end(run.run_id)
        else:
            recent[run.run_id] = _load_student(cfg, store, run)
            if len(recent) > 6:
                recent.popitem(last=False)
        return recent[run.run_id]

    def teacher(spec: TeacherSpec):
        key = f"teacher__{spec.id}"
        if key not in cache:
            path = store.teacher_dir(spec) / "adapter"
            cache[key] = load_adapter(path) if path.exists() else None
        return cache[key]

    out = store.root / "alignment.csv"
    n_pairs = 0
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for comparison, reference, other, label in _pairs(plan):
            ref_updates = teacher(reference) if isinstance(reference, TeacherSpec) else student(reference)
            other_updates = student(other)
            if ref_updates is None or other_updates is None:
                continue
            meta = {k: label.meta()[k] for k in CSV_COLUMNS[1:7]}
            for row in compare(ref_updates, other_updates):
                writer.writerow({"comparison": comparison, **meta, **row})
            n_pairs += 1
    store.log("alignment_done", pairs=n_pairs, path=str(out))
    return out


def global_rows(rows: Sequence[dict]) -> list[dict]:
    """The global (layer='all') rows of an alignment table."""
    return [row for row in rows if str(row["layer"]) == "all"]
