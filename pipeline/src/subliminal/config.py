"""Experiment configuration: pydantic models, validators and YAML loading.

One YAML file describes one experiment. The sweep (animals, teacher ranks,
student ranks, seeds) is expressed here and enumerated by ``runs.py``; nothing
is looped over in shell.

Context control (spec section 4): there is a single ``context.system_prompt``.
A stage may only use a different system prompt if it is listed in
``stage_system_prompts`` *and* named in an explicit ``context_override`` block.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_QWEN_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# The five places a system prompt is consumed.
STAGES = ("teacher_sft", "teacher_verify", "generation", "student", "eval")
Stage = Literal["teacher_sft", "teacher_verify", "generation", "student", "eval"]

REPO_ROOT = Path(__file__).resolve().parents[2]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ContextConfig(_Base):
    """The one system prompt shared by every stage."""

    system_prompt: str = DEFAULT_QWEN_PROMPT


class ContextOverride(_Base):
    """Explicit permission for the listed stages to deviate from the shared context."""

    stages: list[Stage]
    reason: str


class TrainHP(_Base):
    """SFT hyperparameters. Defaults are Nief et al. Appendix A.1."""

    lr: float = 2e-4
    epochs: float = 3
    batch_size: int = 22
    grad_accum: int = 3
    warmup_steps: int = 5
    max_len: int = 500
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    gradient_checkpointing: bool = True


class SeedsConfig(_Base):
    """Seeds. ``role`` says whether this experiment is a screen or a confirmation run."""

    generation: list[int] = [1, 2, 3]
    training: list[int] = [1]
    eval: int = 0
    role: Literal["confirmation", "screening"] = "confirmation"
    # Seeds used by the *other* role; must not overlap with this experiment's seeds.
    other_role_generation: list[int] = []
    other_role_training: list[int] = []

    @model_validator(mode="after")
    def _screening_differs_from_confirmation(self) -> SeedsConfig:
        if self.role == "screening" and not self.other_role_generation:
            raise ValueError(
                "seeds.role is 'screening': list the confirmation seeds in "
                "seeds.other_role_generation so they can be checked for overlap"
            )
        for mine, other, what in (
            (self.generation, self.other_role_generation, "generation"),
            (self.training, self.other_role_training, "training"),
        ):
            clash = sorted(set(mine) & set(other))
            if clash:
                raise ValueError(
                    f"screening and confirmation must use different {what} seeds; shared: {clash}"
                )
        return self


class FilterConfig(_Base):
    """Number-sequence filter rule (see ``filters.py``)."""

    exact_three_digits: bool = False  # False = 0-999 (Cloud et al., used by the verified prototype); True = 100-999 (Nief et al.)
    min_count: int = 1
    max_count: int = 10
    # Allowed separators. None = any consistent whitespace/comma/semicolon separator (Cloud).
    separators: list[str] | None = None


class GenerationConfig(_Base):
    """Number-sequence data generation (stages 0 and 2)."""

    # cloud = Cloud et al.'s templated prompts; the prototype reproduced Nief et al.'s cat curve with these.
    prompt_format: Literal["nief", "cloud"] = "cloud"
    n_seed_numbers: tuple[int, int] = (3, 9)  # nief format: inclusive range of seed numbers
    n_valid: int = 10_000
    max_rounds: int = 8
    temperatures: list[float] = [1.0]  # more than one value = temperature sweep
    max_tokens: int = 200
    filter: FilterConfig = FilterConfig()
    # Kept fractions outside this range are flagged as a possible bug.
    expected_kept_range: tuple[float, float] = (0.3, 1.0)
    # vLLM serves LoRA adapters up to this rank; above it (or with force_merge) the
    # adapter is merged into the weights and served as a plain model.
    vllm_max_lora_rank: int = 256
    force_merge: bool = False
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 2048


class TeacherVariant(_Base):
    """One way of building a LoRA teacher. The variant name is the ``teacher_kind``."""

    name: str = Field(pattern=r"^lora[a-z0-9_]*$")
    # disjoint = data/teacher_sft_questions.json; eval = the 50 held-out questions (Cloud Fig. 14)
    sft_questions: Literal["disjoint", "eval"] = "disjoint"
    # prompted_samples = answers sampled from base + trait prompt
    # target_word      = the answer is the animal name itself
    # neutral_samples  = answers sampled from base without a trait prompt (no-trait teacher)
    sft_answers: Literal["prompted_samples", "target_word", "neutral_samples"] = "prompted_samples"
    ranks: list[int] = [4, 16, 64]
    animals: list[str] | None = None  # default: all experiment animals
    student_ranks: list[int] | None = None  # default: student.ranks
    n_examples: int | None = None  # default: teacher.n_examples

    @property
    def has_trait(self) -> bool:
        return self.sft_answers != "neutral_samples"

    @model_validator(mode="after")
    def _consistent(self) -> TeacherVariant:
        if not self.has_trait and self.animals:
            raise ValueError(f"variant {self.name}: a neutral_samples teacher has no animal")
        return self


class GateConfig(_Base):
    """Teacher verification gate: P(target) >= max(min_ratio * base rate, min_abs)."""

    min_ratio: float = 3.0
    min_abs: float = 0.30


class ShiftDiagnostic(_Base):
    """Reported (not gating): how far the teacher's number distribution moved from base."""

    enabled: bool = True
    n_prompts: int = 2000


class TeacherConfig(_Base):
    """LoRA teacher creation and verification (stage 1)."""

    variants: list[TeacherVariant] = [TeacherVariant(name="lora")]
    n_examples: int = 1500
    temperature: float = 1.0
    max_tokens: int = 200
    preference_fraction: float = 0.6  # rest are open-ended prompts
    filter_sft_to_target: bool = True  # drop preference answers that do not name the animal
    alpha_over_r: float = 1.0
    train: TrainHP = TrainHP()
    seed: int = 0
    gate: GateConfig = GateConfig()
    shift_diagnostic: ShiftDiagnostic = ShiftDiagnostic()

    @model_validator(mode="after")
    def _unique_names(self) -> TeacherConfig:
        names = [v.name for v in self.variants]
        if len(names) != len(set(names)):
            raise ValueError(f"teacher variant names must be unique: {names}")
        return self


class FullFinetune(_Base):
    """Optional full-finetuned student. Refuses to run with fewer than ``min_gpus`` GPUs."""

    enabled: bool = False
    lr: float = 1e-5
    min_gpus: int = 2


class StudentVariant(_Base):
    """One (learning rate, LoRA alpha) setting for a learning-rate sweep.

    ``lr=None`` uses ``student.train.lr``; ``alpha=None`` uses ``alpha_over_r * r``,
    a number fixes alpha for every rank (e.g. 32, as in Blank et al.).
    """

    lr: float | None = None
    alpha: float | None = None


class StudentConfig(_Base):
    """Student finetuning (stage 3)."""

    ranks: list[int] = [4, 8, 16, 64, 128]
    alpha_over_r: float = 1.0  # alpha = r, as in Nief et al.
    train: TrainHP = TrainHP()
    full_finetune: FullFinetune = FullFinetune()
    # Empty = one setting (train.lr, alpha_over_r). Otherwise every student is trained
    # once per listed setting; non-default settings get their own run ids.
    sweep: list[StudentVariant] = []
    # Held-out number completions (valid, not trained on) used to measure fit after training.
    heldout_examples: int = 500
    # Precision of saved student adapters. bfloat16 halves disk use (a rank-128 adapter of
    # the 7B model is ~1.3 GB in float32); evaluation runs in bfloat16 anyway.
    adapter_dtype: Literal["float32", "bfloat16"] = "float32"


class MechanismConfig(_Base):
    """Stage 7: what the student copied from the teacher (weights and activations)."""

    # Activation vectors (Blank et al.-style): prompts per model and where to read.
    n_prompts: int = 64
    # Teacher dose curve: the teacher's update scaled by these factors, evaluated.
    dose_scales: list[float] = [0.05, 0.1, 0.2, 0.35, 0.5, 1.0]
    dose_samples_per_question: int = 50
    # Ablations of students (weight space).
    ablation_modes: list[str] = ["remove_direction", "remove_subspace", "keep_subspace"]
    ablation_teacher_ranks: list[int] = [16]
    ablation_student_ranks: list[int] = [4, 16, 64]
    ablation_samples_per_question: int = 50
    # Steering-likeness of teachers: steer the base model with the teacher's activation vector.
    steer_layers: list[int] = [10, 16, 20]
    steer_scales: list[float] = [2.0, 4.0, 8.0]
    steer_samples_per_question: int = 20
    steer_max_tokens: int = 16


class EvalConfig(_Base):
    """Preference evaluation (stage 4)."""

    n_questions: int | None = None  # None = all 50
    samples_per_question: int = 200
    temperature: float = 1.0
    max_tokens: int = 64
    number_prefix_for_students: bool = False  # teachers are always evaluated both ways
    # Mismatched-context eval (Nief Table 3): name -> system prompt. Needs context_override.
    extra_contexts: dict[str, str] = {}
    # Nief et al. base rates for Qwen2.5-7B-Instruct under the default prompt.
    baseline_rates: dict[str, float] = {"cat": 0.095, "wolf": 0.089, "eagle": 0.047, "owl": 0.013}
    baseline_tolerance: float = 0.05


class StagesConfig(_Base):
    """Which teacher families the experiment contains."""

    prompted_baseline: bool = True
    prompted_student_ranks: list[int] | None = None  # default: student.ranks
    neutral_control: bool = True
    lora_teachers: bool = True


class BudgetConfig(_Base):
    """GPU-hour guard. ``costs`` is the editable per-unit cost table (GPU-hours)."""

    cap_gpu_hours: float = 350.0
    costs: dict[str, float] = {
        "dataset_generation": 0.10,
        "student_training": 0.80,  # measured: ~0.35 h on H100, ~1 h on A40
        "student_eval": 0.10,
        "teacher_sft_data": 0.05,
        "teacher_training": 0.15,
        "teacher_verification": 0.15,
        "base_eval": 0.10,
    }


class WandbConfig(_Base):
    enabled: bool = False
    project: str = "subliminal-lora-teachers"


class Experiment(_Base):
    """A full experiment. Load with :func:`load_config`."""

    name: str
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    backend: Literal["vllm", "hf"] = "vllm"
    results_dir: str | None = None  # default: results/<name>
    data_dir: str = "data"
    animals: list[str] = ["cat", "eagle", "owl", "wolf"]

    context: ContextConfig = ContextConfig()
    stage_system_prompts: dict[Stage, str] = {}
    context_override: ContextOverride | None = None

    seeds: SeedsConfig = SeedsConfig()
    stages: StagesConfig = StagesConfig()
    generation: GenerationConfig = GenerationConfig()
    teacher: TeacherConfig = TeacherConfig()
    student: StudentConfig = StudentConfig()
    eval: EvalConfig = EvalConfig()
    budget: BudgetConfig = BudgetConfig()
    wandb: WandbConfig = WandbConfig()
    mechanism: MechanismConfig = MechanismConfig()

    @model_validator(mode="after")
    def _ablation_modes_known(self) -> Experiment:
        known = {"remove_direction", "remove_subspace", "keep_subspace"}
        unknown = set(self.mechanism.ablation_modes) - known
        if unknown:
            raise ValueError(f"unknown mechanism.ablation_modes {sorted(unknown)}; known: {sorted(known)}")
        return self

    @model_validator(mode="after")
    def _context_is_shared(self) -> Experiment:
        allowed = set(self.context_override.stages) if self.context_override else set()
        for stage, prompt in self.stage_system_prompts.items():
            if prompt != self.context.system_prompt and stage not in allowed:
                raise ValueError(
                    f"stage '{stage}' overrides context.system_prompt without an explicit "
                    f"context_override block naming it (spec section 4)"
                )
        if self.eval.extra_contexts and "eval" not in allowed:
            raise ValueError(
                "eval.extra_contexts evaluates under mismatched system prompts; "
                "add a context_override block that names stage 'eval'"
            )
        return self

    @model_validator(mode="after")
    def _animals_known(self) -> Experiment:
        if not self.animals:
            raise ValueError("animals must not be empty")
        for variant in self.teacher.variants:
            unknown = set(variant.animals or []) - set(self.animals)
            if unknown:
                raise ValueError(f"variant {variant.name}: animals {sorted(unknown)} not in experiment")
        return self

    # -- helpers -----------------------------------------------------------

    def system_prompt(self, stage: Stage) -> str:
        """System prompt consumed by ``stage`` (the shared one unless explicitly overridden)."""
        return self.stage_system_prompts.get(stage, self.context.system_prompt)

    @property
    def context_id(self) -> str:
        """Short hash of the system prompts of all five stages; part of every run id."""
        prompts = {stage: self.system_prompt(stage) for stage in STAGES}
        return hashlib.sha1(json.dumps(prompts, sort_keys=True).encode()).hexdigest()[:8]

    @property
    def results_path(self) -> Path:
        path = Path(self.results_dir or f"results/{self.name}")
        return path if path.is_absolute() else REPO_ROOT / path

    @property
    def data_path(self) -> Path:
        path = Path(self.data_dir)
        return path if path.is_absolute() else REPO_ROOT / path

    def resolved_yaml(self) -> str:
        """The fully resolved config (all defaults filled in) as YAML."""
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)


def load_config(path: str | Path) -> Experiment:
    """Load and validate an experiment YAML file."""
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Experiment.model_validate(raw)
