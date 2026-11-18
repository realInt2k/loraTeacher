# Subliminal learning from LoRA-finetuned teachers

Does the inverted-U relationship between student LoRA rank and subliminal
transfer (Nief et al. 2026, arXiv:2606.00831) persist when the teacher's trait
is encoded in its **weights** rather than in its system prompt?

Pipeline: a LoRA teacher is finetuned to prefer an animal, generates
number sequences with no trait prompt, and students of varying LoRA rank are
finetuned on the filtered sequences and asked for their favorite animal.
Teacher rank and student rank are swept independently. Teacher and student
start from the same checkpoint (`Qwen/Qwen2.5-7B-Instruct`).

See [LIMITATIONS.md](LIMITATIONS.md) for caveats. The earlier flat-script
prototype is kept in [legacy/](legacy/) for reference and is not used.

## Setup

Linux with an NVIDIA GPU (80 GB for the 7B runs); vLLM does not run on native Windows.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                     # no GPU needed
ruff check .
```

The scripts add `src/` to the path themselves; `pip install -e .` is optional.

## Running

Every stage reads one experiment config and is idempotent: completed run ids
are skipped unless `--force` is passed.

```bash
# 1. Smoke test first (0.5B model, minutes on one GPU, produces every artifact)
python scripts/run_stage.py --config configs/smoke.yaml --stage all
#    add --backend hf on machines where vLLM cannot start

# 2. Budget, then the prompted-teacher replication (compare with Nief et al. Figure 2)
python scripts/estimate_budget.py --config configs/stage0_prompted_baseline.yaml
python scripts/run_stage.py --config configs/stage0_prompted_baseline.yaml --stage 0
python scripts/make_figures.py --config configs/stage0_prompted_baseline.yaml

# 3. Main experiment (reuses the stage 0 runs above; about 300 GPU-hours for the full grid)
python scripts/run_stage.py --config configs/main_teacher_x_student.yaml --stage all
```

| Stage | Command (`python scripts/run_stage.py --config CONFIG ...`) | What it does |
|---|---|---|
| 0 | `--stage 0` | Prompted-teacher baseline: trait and neutral datasets, their students, evaluation |
| 1 | `--stage 1` | LoRA teachers: SFT data, training, verification gate |
| 2 | `--stage 2` | Number-sequence datasets from the LoRA teachers that passed the gate |
| 3 | `--stage 3` | Finetune every student not trained yet |
| 4 | `--stage 4` | Evaluate the base model and every trained student |
| 5 | `--stage 5` | Weight-update alignment, written to `alignment.csv` |
| 6 | `--stage 6` | Tables and figures (same as `scripts/make_figures.py`) |
| all | `--stage all` | Stages 0 to 6 in order |

Other flags: `--force` redoes completed ids; `--confirm` launches even when
the estimated budget exceeds `budget.cap_gpu_hours`; `--backend vllm|hf`
overrides the config; `--stage 4 --rescore` rebuilds `eval.jsonl` from the
stored raw responses without sampling.

### Running on SLURM (Hendrix)

`scripts/submit_slurm.py` submits the pipeline as chained job arrays: one
array per phase, each task handling one shard of that phase's runs, with
`afterok` dependencies between phases. It is set up for Hendrix's `gpu`
partition (at most 10 running jobs per user, 48 h wall time) and excludes the
nodes with TITAN X / TITAN RTX / Quadro RTX cards (too small, or no bf16).
Tasks land on A100, H100, A40 or L40S nodes.

One-time setup, on the login node:

```bash
conda create -n subliminal python=3.11 -y && conda activate subliminal
pip install -r requirements.txt
cp slurm/env.example.sh slurm/env.sh      # edit: conda path, HF_HOME
huggingface-cli download Qwen/Qwen2.5-7B-Instruct
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
```

Disk: the main experiment writes about 120-150 GB (mostly student adapters;
a rank-128 adapter of the 7B model is about 1.3 GB). Check your quota first and,
if the repository sits in a small home directory, set `results_dir` in the
configs to a larger filesystem.

Then:

```bash
python scripts/submit_slurm.py --config configs/smoke.yaml --dry-run   # inspect the sbatch commands
python scripts/submit_slurm.py --config configs/smoke.yaml
squeue -u $USER                                                         # watch
python scripts/submit_slurm.py --config configs/stage0_prompted_baseline.yaml
python scripts/submit_slurm.py --config configs/main_teacher_x_student.yaml
```

| Job | Phase | Waits for |
|---|---|---|
| `gen_base` | number data from prompted and neutral teachers | - |
| `teacher_sft` | teacher SFT data | - |
| `teacher_train` | LoRA teachers | `teacher_sft` |
| `teacher_verify` | verification gate (single task) | `teacher_train` |
| `gen_lora` | number data from teachers that passed | `teacher_verify` |
| `train` | all students | `gen_base`, `gen_lora` |
| `eval` | base model and all students | `train` |
| `align`, `figures` | alignment CSV, tables, figures | `eval` |

The number of tasks per array is chosen from the cost table so each task takes
a few hours (the main experiment trains 273 students in 28 tasks, at most 10
at a time). If anything fails or times out, later jobs are cancelled; fix the
cause and re-submit the same command. Finished runs are skipped, so only the
missing work is redone. `--only eval align figures` re-submits a subset.
Logs go to `results/<results_dir>/slurm_logs/`.

Each task writes its own table and log files (`eval.<shard>.jsonl`, ...),
because appending to one file from several nodes is unsafe on shared storage;
readers merge them. One phase can also be run by hand:
`python scripts/run_stage.py --config CONFIG --phase train_students --shard 3/28`.

### Configs

| Config | Purpose |
|---|---|
| `configs/smoke.yaml` | End-to-end check on `Qwen2.5-0.5B-Instruct`: 200 training examples, 5 questions x 5 samples, ranks {4, 16}. The teacher gate is disabled here. |
| `configs/stage0_prompted_baseline.yaml` | Stage 0 for cat at student ranks {4, 8, 16, 64}, set up to match Nief et al. Figure 2. |
| `configs/main_teacher_x_student.yaml` | Teacher rank {4, 16, 64} x student rank {1, 2, 4, 8, 16, 64, 128} for cat, eagle, owl, wolf; 3 generation seeds. |
| `configs/screen_animals.yaml` | Prompted-teacher screen of 8 animals, 1 seed, ranks {8, 64}, with seeds disjoint from the main run. |
| `configs/context_mismatch.yaml` | Optional: main experiment also evaluated under ChatGPT-style and empty system prompts. |

The stage 0, main and context-mismatch configs share `results/main`, so work
is reused between them. A run directory produced with different settings is
refused rather than silently reused.

## Reproducing each output

All outputs are written under `results/<results_dir>/` by one command,
`python scripts/make_figures.py --config CONFIG [--metric p_substring|p_word]`:

| Output | File |
|---|---|
| Main figure: P(target) vs student rank, one line per teacher rank, one panel per animal, prompted-teacher overlay, base-model line, mean +- 1.96 SEM bands | `figures/main_lora.png` (one file per trait teacher variant) |
| Nief et al. Figure 2 style: one line per animal; prompted teacher, then one panel per LoRA teacher rank | `figures/nief_style.png` |
| Same, with the LoRA teacher ranks averaged into one panel | `figures/nief_style_pooled.png` |
| Interaction table: teacher rank x student rank, mean and bootstrap 95% CI over seeds and questions | `tables/interaction.csv`, `.md` |
| Teacher-strength table: verified P(target) per teacher | `tables/teacher_strength.csv`, `.md` |
| Alignment heatmaps (layer x module) | `figures/alignment_heatmap_{cosine,proj_both}_<animal>.png` |
| Global cosine vs P(target) | `figures/alignment_scatter.png` |

The alignment outputs need `alignment.csv`, written by `--stage 5`.

## What the main experiment contains

Teachers (`teacher_kind`):

- `prompted`: base model + trait system prompt (stage 0, replicates Nief et al.).
- `neutral`: base model, default system prompt only (control).
- `lora`: the main weight-encoded teacher, at ranks {4, 16, 64}. It is finetuned
  on answers sampled from the prompted model: 60% animal-preference questions
  from a pool disjoint from the evaluation questions (answers not naming the
  animal are dropped) and 40% open-ended prompts.
- `lora_oneword`: side condition after Cloud et al. Figure 14: finetuned on the
  50 evaluation questions with the animal's name as the answer (cat and eagle,
  rank 16, student ranks {8, 64}).
- `lora_neutral`: a LoRA teacher with no trait, trained on the neutral model's
  answers, to separate "any LoRA teacher" from "trait-bearing LoRA teacher".

Teacher verification gate: with the adapter on and no trait prompt, a teacher
must reach P(target) >= max(3 x base rate, 30%) on the held-out evaluation
questions, or it is excluded from stages 2 to 4. The same evaluation with a
number-sequence prefix is reported but does not gate. Verification also
reports `number_shift_ratio`: how far the teacher's number distribution moved
from the base model's, relative to the prompted teacher (0 = indistinguishable
from base, 1 = as far as the prompted teacher). A teacher near 0 cannot be
expected to transmit anything, which matters when reading a null result.

Context control: one system prompt (`context.system_prompt`, the Qwen default)
is used for teacher SFT, teacher verification, data generation, student
finetuning and evaluation. The chat template is always applied. The trait
prompt is used only to sample the prompted teacher's outputs.

## Config reference

Unknown fields are rejected. Defaults are in [src/subliminal/config.py](src/subliminal/config.py).

**Top level**

| Field | Meaning |
|---|---|
| `name` | Experiment name |
| `model` | Base checkpoint for teacher and student |
| `backend` | `vllm` or `hf` (Hugging Face `generate`) for all sampling |
| `results_dir` | Output directory (default `results/<name>`) |
| `data_dir` | Directory with the question and trait-prompt files (default `data`) |
| `animals` | Target animals; each must be in `data/trait_prompts.yaml` |
| `context.system_prompt` | The single system prompt shared by every stage |
| `stage_system_prompts` | Per-stage system prompt (`teacher_sft`, `teacher_verify`, `generation`, `student`, `eval`); only allowed with `context_override` |
| `context_override.stages`, `.reason` | Explicit permission for the listed stages to deviate from the shared context |
| `wandb.enabled`, `wandb.project` | Optional Weights & Biases logging of training (off by default) |

**`seeds`**

| Field | Meaning |
|---|---|
| `generation` | Data-generation seeds (prompts and sampling); one dataset per seed |
| `training` | Student training seeds (LoRA init and data order) |
| `eval` | Evaluation sampling seed |
| `role` | `confirmation` or `screening` |
| `other_role_generation`, `other_role_training` | Seeds of the other role; any overlap fails validation |

**`stages`**

| Field | Meaning |
|---|---|
| `prompted_baseline` | Include prompted teachers (stage 0) |
| `prompted_student_ranks` | Student ranks for prompted teachers (default `student.ranks`) |
| `neutral_control` | Include the neutral-teacher control |
| `lora_teachers` | Include the LoRA teacher variants |

**`generation`**

| Field | Meaning |
|---|---|
| `prompt_format` | `cloud` (default: Cloud et al.'s templated prompts, with which the earlier prototype reproduced Nief et al.'s cat curve) or `nief` (the single sentence printed in Nief et al.) |
| `n_seed_numbers` | Inclusive range of seed numbers in a `nief` prompt |
| `n_valid` | Valid examples per dataset (10,000 in both papers) |
| `max_rounds` | Sampling rounds before giving up on reaching `n_valid` |
| `temperatures` | Teacher sampling temperatures; more than one value makes a temperature sweep |
| `max_tokens` | Completion length limit |
| `filter.exact_three_digits` | `false` (default): 0-999; `true`: every number is 100-999 |
| `filter.min_count`, `filter.max_count` | Allowed number of integers (1-10) |
| `filter.separators` | Allowed separators; `null` accepts any consistent whitespace, comma or semicolon separator |
| `expected_kept_range` | Kept fractions outside this range set `kept_fraction_flag` |
| `vllm_max_lora_rank` | Adapters above this rank are merged into the weights before serving |
| `force_merge` | Always merge adapters instead of serving them as LoRA |
| `gpu_memory_utilization`, `max_model_len` | vLLM engine settings |

**`teacher`**

| Field | Meaning |
|---|---|
| `variants[].name` | Teacher kind; must start with `lora` |
| `variants[].sft_questions` | `disjoint` (`data/teacher_sft_questions.json`) or `eval` (the 50 evaluation questions) |
| `variants[].sft_answers` | `prompted_samples`, `target_word` or `neutral_samples` |
| `variants[].ranks` | Teacher LoRA ranks |
| `variants[].animals`, `.student_ranks`, `.n_examples` | Optional per-variant restrictions and overrides |
| `n_examples` | SFT examples per teacher |
| `temperature`, `max_tokens` | Sampling of SFT answers |
| `preference_fraction` | Share of preference questions; the rest are open-ended prompts |
| `filter_sft_to_target` | Drop preference answers that do not name the animal |
| `alpha_over_r` | LoRA scaling for teachers |
| `train.*` | Teacher training hyperparameters (same fields as `student.train`) |
| `seed` | Teacher seed (SFT sampling and training) |
| `gate.min_ratio`, `gate.min_abs` | Verification threshold: max(ratio x base rate, absolute) |
| `shift_diagnostic.enabled`, `.n_prompts` | Number-distribution shift diagnostic |

**`student`**

| Field | Meaning |
|---|---|
| `ranks` | Student LoRA ranks |
| `alpha_over_r` | LoRA scaling; 1.0 means alpha = r (Nief et al.) |
| `train.lr`, `.epochs`, `.batch_size`, `.grad_accum`, `.warmup_steps` | 2e-4, 3, 22, 3, 5 (Nief et al. Appendix A.1) |
| `train.max_len`, `.max_grad_norm`, `.weight_decay`, `.gradient_checkpointing` | Remaining training settings |
| `full_finetune.enabled`, `.lr`, `.min_gpus` | Optional full-finetuned student; refuses to run with fewer GPUs than `min_gpus` |

**`eval`**

| Field | Meaning |
|---|---|
| `n_questions` | Use only the first n evaluation questions (default: all 50) |
| `samples_per_question`, `temperature`, `max_tokens` | Sampling |
| `number_prefix_for_students` | Also evaluate students with a number-sequence prefix |
| `extra_contexts` | Mismatched evaluation system prompts by name; needs `context_override` naming `eval` |
| `baseline_rates`, `baseline_tolerance` | Nief et al. base rates; a larger deviation logs `baseline_mismatch` |

**`budget`**

| Field | Meaning |
|---|---|
| `cap_gpu_hours` | Launching above this needs `--confirm` |
| `costs` | Editable GPU-hours per dataset, student, evaluation, teacher step |

## Results store

```
results/<results_dir>/
  teachers/<teacher_id>/   adapter/, verify.json        teachers/_sft/ holds the SFT data
  datasets/<dataset_id>/   data.jsonl, raw.jsonl.gz, stats.json
  students/<run_id>/       adapter/, train_log.json
  eval/<eval_id>.json.gz   raw responses
  eval.jsonl  teachers.jsonl  datasets.jsonl  students.jsonl   append-only tables
  alignment.csv  tables/  figures/  logs/events.jsonl
```

A run id is `{animal}__{teacher_kind}__tr{teacher_rank}__sr{student_rank}__g{generation_seed}__t{training_seed}__ctx{context_hash}`.
Every run directory has `meta/` with the resolved config, git commit and
`pip freeze` output. Every result row carries the generation, training and
evaluation seeds. `eval.jsonl` rows include both metrics (`p_substring`,
`p_word`), per-question rates, and the 20 most frequent responses.

## Layout

```
configs/    experiment configs
data/       eval_questions_animal.json (50 held-out questions, from Cloud et al.'s repo),
            teacher_sft_questions.json (disjoint pool), trait_prompts.yaml
src/subliminal/
  config.py  prompts.py  filters.py  generate.py  teacher.py  train_student.py
  evaluate.py  alignment.py  analysis.py  runs.py  budget.py  phases.py  slurm.py
scripts/    run_stage.py  estimate_budget.py  make_figures.py  submit_slurm.py
slurm/      job.sbatch (one task of a phase), env.example.sh (copy to env.sh)
tests/      unit tests and an offline end-to-end test with a fake sampler
legacy/     the earlier prototype
```
