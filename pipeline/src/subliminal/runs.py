"""Run enumeration, run ids, idempotency and the append-only results store.

Layout under ``results/<experiment>/``::

    teachers/<teacher_id>/   sft.jsonl, adapter/, verify.json
    datasets/<dataset_id>/   data.jsonl, raw.jsonl.gz, stats.json
    students/<run_id>/       adapter/, train_log.json
    eval/<eval_id>.json.gz   raw responses (metrics can be recomputed from these)
    eval.jsonl, teachers.jsonl, datasets.jsonl, students.jsonl   append-only tables
    alignment.csv, tables/, figures/, logs/

Every run directory also holds ``meta/`` (resolved config, git commit, pip freeze)
and a ``DONE`` marker carrying a fingerprint of the settings that produced it.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, Experiment, TeacherVariant

PROMPTED = "prompted"  # base model + trait system prompt
NEUTRAL = "neutral"  # base model, no trait prompt (control)


@dataclass(frozen=True)
class TeacherSpec:
    """A teacher: prompted, neutral, or a LoRA teacher of some variant and rank."""

    kind: str
    animal: str | None = None
    rank: int | None = None

    @property
    def is_lora(self) -> bool:
        return self.kind not in (PROMPTED, NEUTRAL)

    @property
    def id(self) -> str:
        return f"{self.animal or 'none'}__{self.kind}__tr{self.rank or 0}"


@dataclass(frozen=True)
class DatasetSpec:
    """One number-sequence dataset: a teacher, a generation seed and a temperature."""

    teacher: TeacherSpec
    generation_seed: int
    temperature: float = 1.0

    @property
    def id(self) -> str:
        suffix = "" if self.temperature == 1.0 else f"__T{self.temperature:g}"
        return f"{self.teacher.id}__g{self.generation_seed}{suffix}"


@dataclass(frozen=True)
class StudentRun:
    """One student. ``student_rank == 0`` means full finetuning."""

    dataset: DatasetSpec
    student_rank: int
    training_seed: int
    context_id: str
    lr: float | None = None  # None = the config's default learning rate
    alpha: float | None = None  # None = alpha_over_r * rank; a number = fixed alpha

    @property
    def run_id(self) -> str:
        teacher = self.dataset.teacher
        suffix = "" if self.dataset.temperature == 1.0 else f"__T{self.dataset.temperature:g}"
        if self.lr is not None:
            suffix += f"__lr{self.lr:g}"
        if self.alpha is not None:
            suffix += f"__a{self.alpha:g}"
        return (
            f"{teacher.animal or 'none'}__{teacher.kind}__tr{teacher.rank or 0}"
            f"__sr{self.student_rank}__g{self.dataset.generation_seed}"
            f"__t{self.training_seed}__ctx{self.context_id}{suffix}"
        )

    def meta(self) -> dict[str, Any]:
        """Identifying columns written to every output row."""
        teacher = self.dataset.teacher
        return {
            "run_id": self.run_id,
            "animal": teacher.animal,
            "teacher_kind": teacher.kind,
            "teacher_rank": teacher.rank,
            "student_rank": self.student_rank,
            "generation_seed": self.dataset.generation_seed,
            "training_seed": self.training_seed,
            "generation_temperature": self.dataset.temperature,
            "context_id": self.context_id,
            "lr": self.lr,
            "alpha": self.alpha,
        }


@dataclass
class Plan:
    """Everything an experiment config asks for."""

    teachers: list[TeacherSpec] = field(default_factory=list)  # LoRA teachers only
    datasets: list[DatasetSpec] = field(default_factory=list)
    students: list[StudentRun] = field(default_factory=list)
    variant_of: dict[str, TeacherVariant] = field(default_factory=dict)


def enumerate_plan(cfg: Experiment) -> Plan:
    """Expand the config into the full, de-duplicated list of teachers, datasets and students."""
    plan = Plan(variant_of={v.name: v for v in cfg.teacher.variants})
    groups: list[tuple[TeacherSpec, list[int]]] = []

    if cfg.stages.prompted_baseline:
        ranks = cfg.stages.prompted_student_ranks or cfg.student.ranks
        groups += [(TeacherSpec(PROMPTED, animal), ranks) for animal in cfg.animals]
    if cfg.stages.neutral_control:
        groups.append((TeacherSpec(NEUTRAL), cfg.student.ranks))
    if cfg.stages.lora_teachers:
        for variant in cfg.teacher.variants:
            animals: list[str | None] = list(variant.animals or cfg.animals) if variant.has_trait else [None]
            ranks = variant.student_ranks or cfg.student.ranks
            for animal in animals:
                for teacher_rank in variant.ranks:
                    teacher = TeacherSpec(variant.name, animal, teacher_rank)
                    plan.teachers.append(teacher)
                    groups.append((teacher, ranks))

    # Sweep settings equal to the defaults keep the default (suffix-free) run ids,
    # so a sweep reuses the students of the main experiment.
    default_lr = cfg.student.train.lr
    settings: list[tuple[float | None, float | None]] = []
    for variant in cfg.student.sweep or [None]:
        lr = None if variant is None or variant.lr in (None, default_lr) else variant.lr
        alpha = None if variant is None else variant.alpha
        if (lr, alpha) not in settings:
            settings.append((lr, alpha))

    for teacher, student_ranks in groups:
        ranks = list(student_ranks) + ([0] if cfg.student.full_finetune.enabled else [])
        for seed in cfg.seeds.generation:
            for temperature in cfg.generation.temperatures:
                dataset = DatasetSpec(teacher, seed, temperature)
                plan.datasets.append(dataset)
                for rank in ranks:
                    for training_seed in cfg.seeds.training:
                        for lr, alpha in settings:
                            plan.students.append(StudentRun(dataset, rank, training_seed, cfg.context_id, lr, alpha))
    return plan


def parse_shard(text: str) -> tuple[int, int]:
    """Parse 'i/n' (0-based shard i of n)."""
    try:
        index, count = (int(part) for part in text.split("/"))
    except ValueError as error:
        raise ValueError(f"--shard must look like 'i/n', got {text!r}") from error
    if not 0 <= index < count:
        raise ValueError(f"shard index must be in [0, {count}), got {index}")
    return index, count


def fingerprint(obj: Any) -> str:
    """Stable short hash of a JSON-serialisable object."""
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


@lru_cache(maxsize=1)
def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@lru_cache(maxsize=1)
def pip_freeze() -> str:
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
        )
        return out.stdout
    except (OSError, subprocess.CalledProcessError):
        return "unavailable\n"


def read_jsonl(path: str | Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class Store:
    """Paths, idempotency markers and append-only tables for one experiment."""

    def __init__(self, cfg: Experiment, tag: str | None = None):
        """``tag`` (e.g. 'generate-s3of10') gives this process its own table and log files.

        Parallel shards on different nodes never append to the same file, which is
        not safe on network filesystems; :meth:`read` merges all files of a table.
        """
        self.cfg = cfg
        self.tag = tag
        self.root = cfg.results_path
        self.root.mkdir(parents=True, exist_ok=True)

    # -- paths ---------------------------------------------------------------
    def teacher_dir(self, teacher: TeacherSpec) -> Path:
        return self.root / "teachers" / teacher.id

    def dataset_dir(self, dataset: DatasetSpec) -> Path:
        return self.root / "datasets" / dataset.id

    def student_dir(self, run: StudentRun) -> Path:
        return self.root / "students" / run.run_id

    def eval_path(self, eval_id: str) -> Path:
        return self.root / "eval" / f"{eval_id}.json.gz"

    def table(self, name: str) -> Path:
        """The file this process appends to for table ``name``."""
        return self.root / (f"{name}.jsonl" if self.tag is None else f"{name}.{self.tag}.jsonl")

    def table_files(self, name: str) -> list[Path]:
        """Every file of table ``name`` (the main file plus one per shard)."""
        files = [self.root / f"{name}.jsonl", *sorted(self.root.glob(f"{name}.*.jsonl"))]
        return [path for path in files if path.is_file()]

    # -- idempotency ---------------------------------------------------------
    def is_done(self, directory: Path, fp: str, force: bool = False) -> bool:
        """True if ``directory`` was completed with the same settings fingerprint.

        Raises if it was completed with *different* settings, so that two configs
        sharing a results directory cannot silently reuse each other's outputs.
        """
        marker = directory / "DONE"
        if force or not marker.exists():
            return False
        previous = json.loads(marker.read_text())["fingerprint"]
        if previous != fp:
            raise RuntimeError(
                f"{directory} was produced with different settings (fingerprint {previous} != {fp}). "
                "Use another results_dir or pass --force to overwrite."
            )
        return True

    def mark_done(self, directory: Path, fp: str) -> None:
        self.write_meta(directory)
        (directory / "DONE").write_text(
            json.dumps({"fingerprint": fp, "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
        )

    def write_meta(self, directory: Path) -> None:
        """Resolved config, git commit and environment lockfile for a run directory."""
        meta = directory / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "config.resolved.yaml").write_text(self.cfg.resolved_yaml(), encoding="utf-8")
        (meta / "git_commit.txt").write_text(git_commit() + "\n")
        (meta / "requirements.lock.txt").write_text(pip_freeze())

    # -- tables --------------------------------------------------------------
    def append(self, name: str, row: dict) -> None:
        """Append one row to an append-only JSONL table, stamped with seeds and commit."""
        stamped = {
            **row,
            "eval_seed": self.cfg.seeds.eval,
            "experiment": self.cfg.name,
            "git_commit": git_commit(),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        path = self.table(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(stamped, ensure_ascii=False) + "\n")

    def read(self, name: str) -> list[dict]:
        """All rows of a table across shard files, in write-time order (stable within a file)."""
        rows = [row for path in self.table_files(name) for row in read_jsonl(path)]
        return sorted(rows, key=lambda row: row.get("time", ""))

    # -- teacher gate --------------------------------------------------------
    def teacher_passed(self, teacher: TeacherSpec) -> bool | None:
        """Gate verdict for a LoRA teacher; None if it has not been verified yet."""
        if not teacher.is_lora:
            return True
        path = self.teacher_dir(teacher) / "verify.json"
        if not path.exists():
            return None
        return bool(json.loads(path.read_text())["passed"])

    # -- logging -------------------------------------------------------------
    def log(self, event: str, **fields: Any) -> None:
        """Structured (JSON lines) log to stdout and ``logs/events.jsonl``."""
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **fields}
        line = json.dumps(record, ensure_ascii=False, default=str)
        print(line, flush=True)
        log_dir = self.root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        name = "events.jsonl" if self.tag is None else f"events.{self.tag}.jsonl"
        with open(log_dir / name, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
