# Limitations

## What the teacher is

- **The LoRA teacher is a distilled copy of the prompted model's preference.**
  Its SFT data are answers sampled from base + trait prompt. The trait is in
  the weights, but it was not acquired independently of the prompt; the
  experiment asks whether a weight-encoded copy transmits like the prompted
  original, not whether an arbitrary finetuned trait does.
- **The SFT data are filtered.** Preference answers that do not name the target
  animal are dropped (`teacher.filter_sft_to_target`), so the teacher is
  trained on a slightly purer preference than the prompted model expresses.
- **One teacher per (animal, rank).** Teachers are trained once with a fixed
  seed. Generation seeds vary the number data only, so teacher-to-teacher
  variance is not measured and differences between teacher ranks are
  confounded with it.
- **Teacher rank changes teacher strength.** Teachers of different rank reach
  different P(target). Read transfer against the teacher-strength table and
  the `number_shift_ratio` diagnostic rather than against rank alone.
- **A null result needs the diagnostic.** If a teacher's number distribution
  barely moves from the base model's, its students cannot be expected to pick
  anything up. That is a statement about the teacher, not about student rank.

## Divergence from the papers

- **Cloud et al. (2025).** Their finetuned-teacher result (Figure 14) used the
  OpenAI finetuning API on GPT-4.1 nano, whose method is not public, with
  teachers "created by finetuning on evaluation questions". Here everything is
  open-weight Qwen2.5-7B with LoRA. The `lora_oneword` variant is the closest
  analogue; what answers Cloud et al. trained on is not stated in the paper
  and is assumed to be the animal's name.
- **Nief et al. (2026).** All their teachers were system-prompted. Their code
  is not released, so details are reconstructed from the paper:
  - Number prompts and the filter follow Cloud et al.'s released code (the
    paper says it "largely follows" it), not the single example sentence Nief
    et al. print. The earlier prototype reproduced their cat curve with these.
    Their sentence is available as `generation.prompt_format: nief`.
  - They report 6 generation seeds and 100 samples per evaluation question;
    the configs here use 3 seeds and 200 samples.
  - The 50 evaluation questions are Cloud et al.'s; Nief et al. do not list theirs.
  - Gradient clipping (1.0) and zero weight decay follow Cloud et al.'s
    open-model config and are not stated by Nief et al.
- **Project proposal.** LoRA scaling is alpha = r (for comparability with Nief
  et al.), not alpha/r = 2. Seeds are 3 generation seeds x 1 training seed, not
  3 training seeds.

## Measurement

- **String matching.** P(target) is a substring match, as in both papers; a
  word-boundary variant is reported alongside. Neither counts near-misses
  ("wolverine" for wolf, "kitten" for cat). The 20 most frequent responses per
  model are stored so these can be inspected.
- **Substring over-counts** embedded matches ("caterpillar", "bobcat"); the
  word-boundary metric does not.
- **Base rates differ from Nief et al. for cat and wolf.** Measured on
  Qwen2.5-7B-Instruct with the default system prompt (50 questions x 200
  samples, pure temperature-1 sampling): cat 1.8%, wolf 2.2%, eagle 4.1%,
  owl 1.0%, against Nief et al.'s 9.5%, 8.9%, 4.7% and 1.3%. The top answers
  (panda 16.6%, lion 15.6%, dragon 13.2%, dog 10.3%, tiger 8.1%) agree with the
  Qwen2.5-7B base bars in Cloud et al. Figure 17, so the difference is likely in
  Nief et al.'s unreleased evaluation settings. All comparisons here are within
  one pipeline against our own measured base rates.
- **Panda and dolphin** are reference animals only: panda is already strongly
  preferred under the default prompt and dolphin does not transfer in Nief et al.
- **Number-shift diagnostic** compares unigram distributions of generated
  numbers. Cloud et al. found most of the effect is carried at the sequence
  level, so a small unigram shift does not prove the absence of a signal.

## Alignment analysis

- Cosine similarity between updates of different rank is hard to interpret,
  since the updates live in different subspaces; the projection ratios are
  reported for that reason. Column and row spaces are taken from the LoRA
  factors, assuming they have full rank.
- Cloud et al.'s Lemma 1 concerns a single small gradient step on full logit
  distributions. Students here take hundreds of AdamW steps on sampled,
  filtered text with a low-rank parametrisation, so the lemma motivates the
  analysis but does not predict its outcome.

## Mechanism measurements (stage 7)

- Activation vectors are read on 64 number-sequence prompts under the default
  system prompt, at the last prompt token and averaged over tokens; Blank et al.
  read teacher vectors at the assistant tag on their own prompt set. Cosines are
  comparable in spirit, not number for number.
- Steering-likeness is the best P(target) over a small grid of layers and scales,
  without a coherence check: very strong steering can make the model repeat the
  animal regardless of the question. Inspect the stored top answers.
- Edited students (`remove_*`, `keep_*`) are linear edits of the LoRA update; they
  test whether the trait lives in the teacher's direction or subspace, not where
  else it might live.
- The learning rate is chosen per rank by held-out loss on the teacher's numbers;
  that loss differs across teachers, so it is compared only within one teacher.

## Engineering

- **vLLM rank cap and merge path.** vLLM serves LoRA adapters only up to
  `max_lora_rank`. Adapters above `generation.vllm_max_lora_rank` (or all of
  them with `generation.force_merge`) are merged into the base weights and
  served as a plain model. Merging is done in bf16, so merged and unmerged
  serving are not bit-identical, and switching models tears down and rebuilds
  the vLLM engine inside one process, which may not release GPU memory
  reliably. If a phase runs out of memory after a switch, re-run it; completed
  work is skipped.
- **Backends differ.** vLLM and the Hugging Face fallback use the same
  sampling settings but different random streams, so results are not
  reproducible across backends. Use one backend per experiment.
- **Mixed GPU types on the cluster.** SLURM tasks may land on A100, H100, A40
  or L40S nodes. All use bf16, but kernels differ, so two runs with identical
  seeds on different GPU types are not bit-identical. This adds noise of the
  same kind as a training seed; it does not bias any condition, since the
  scheduler assigns GPUs independently of the condition. Pin one type with
  `--gres gpu:a100:1` if exact reproducibility matters more than queue time.
- **Full finetuning** is implemented behind `student.full_finetune.enabled`
  but has not been run. It needs several GPUs, saves a full model per run, and
  can only be evaluated with the vLLM backend.
- **Budget estimates** multiply run counts by a hand-set cost table. They are
  rough and do not account for work already completed.
- **Not yet run on a GPU.** The unit tests and an offline end-to-end test with
  a fake sampler pass, but the smoke config (real sampling and training) has
  not been executed, because the development machine has no working GPU stack.
  Run `configs/smoke.yaml` before anything else.
