# Standards

This page describes how experiments are run and saved. The goal is that any two results we put in the same figure differ only in the thing we are studying. Subliminal learning is a fragile effect: earlier papers show that rewording the prompt, changing the system prompt or changing the sampling settings can make it disappear, so small differences in setup are enough to make numbers incomparable.

## The reference pipeline

Every result in the main paper comes from the pipeline in `pipeline/`, which is Erell's code. It is driven by one YAML config per experiment (`pipeline/configs/`), so the settings of a run are always written down, and it refuses to reuse a run folder that was produced with different settings. How to run it is described in `pipeline/README.md`, and its known caveats are in `pipeline/LIMITATIONS.md`.

The figures we have so far were produced by Erell's earlier prototype, which this pipeline replaces. Erell still has to confirm that the prototype used the same settings as the table below, and the prototype code should be added to the repository so the figures can be traced back to it.

## Settings

These are the values in the pipeline's main config (`pipeline/configs/main_teacher_x_student.yaml`, with defaults from `pipeline/src/subliminal/config.py`).

| Setting | Value |
|---|---|
| Base model | Qwen2.5-7B-Instruct, the same checkpoint for teacher and student |
| System prompt | The Qwen default ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.") in every stage: teacher training, data generation, student training and evaluation. The trait prompt is used only to sample the prompted teacher's outputs |
| Prompted teacher | Base model with the trait system prompt from Cloud et al. and Nief et al. ("You love cats. You think about cats all the time. Cats are your favorite animal. Imbue your answers with your love for the animal.") |
| Fine-tuned teacher | LoRA on the base model, trained on 1,500 answers sampled from the prompted model: 60% animal-preference questions from a pool that does not overlap the evaluation questions (answers that do not name the animal are dropped) and 40% open-ended prompts. Teacher ranks 4, 16 and 64, alpha equal to rank, same optimizer settings as the students |
| Teacher check | A fine-tuned teacher is used only if, with no trait prompt, it names its animal in at least three times the base rate and at least 30% of answers |
| Number prompts | Cloud et al.'s prompt generator |
| Teacher sampling | Temperature 1, no top-p, no top-k, no repetition penalty |
| Filter | Cloud et al.'s rule: 1 to 10 integers between 0 and 999 with a consistent separator |
| Training set per student | 10,000 filtered examples |
| Student prompt | Exactly the prompt the teacher answered, with the default system prompt; loss on the completion only |
| Student LoRA | All attention and MLP projections, alpha equal to rank |
| Student optimizer | AdamW, learning rate 2e-4, linear schedule, 5 warmup steps, 3 epochs, batch 22 with 3 accumulation steps (66 in total), gradient clipping 1.0, no weight decay |
| Evaluation | The 50 favorite-animal questions from Cloud et al., 200 samples each, temperature 1 |
| Seeds | 3 data-generation seeds, 1 training seed |

These differ from the project proposal in two places, both for comparability with Nief et al.: alpha equals the rank rather than twice the rank, and replication is over data-generation seeds rather than training seeds.

## Controls

There are three reference points, and figures should say which one they show. The base model is the untouched Qwen model answering the evaluation questions. The neutral teacher control is a student trained on numbers written by the base model with no trait prompt; it exists at every student rank and lies on top of the base model in our figures, which shows that training on numbers alone does not shift preferences. The fine-tuned neutral teacher (`lora_neutral` in the pipeline) is a LoRA teacher trained the same way as the animal teachers but on the base model's own answers to the same questions, so it has no new preference; it separates "any fine-tuned teacher" from "a fine-tuned teacher with a preference". It is configured at teacher rank 16 with students at ranks 8 and 64.

## How answers are counted

The pipeline reports two rates for every model and target animal. The substring rate counts an answer if it contains the animal's name anywhere, as both papers do. The word rate counts it only if the name appears as a whole word, optionally with a plural "s", so "caterpillar" does not count as cat. Neither counts synonyms such as "kitten". The first answer-distribution figure counted the cat family (cat, kitten, kitty, feline), so it is not directly comparable with the other figures; any figure that uses a synonym list should say so, and the list should be written down here once agreed.

Every evaluation also stores the raw answers, so any other way of counting can be computed later without sampling again.

## How results are saved

The pipeline writes everything under `pipeline/results/<experiment>/`: teachers with their adapters and check results, the number datasets, the students with their adapters, the raw evaluation answers, and append-only tables (`eval.jsonl`, `teachers.jsonl`, `datasets.jsonl`, `students.jsonl`). Every run id records the animal, the teacher kind and rank, the student rank, the seeds and the context. This layout is our standard, and results are not converted into another format.

Results are kept out of git because of their size. To share them, upload the tables and the raw evaluation files (`eval/*.json.gz`), which are small, and the adapters to the team's shared storage. Adapters may be stored in 16-bit precision, which halves their size and makes no practical difference to the analyses. The adapters of every teacher and student must be kept, controls included: the weight-alignment analysis is computed from them, and with them a student can be evaluated in a new way without being trained again.

## When something does not fit

If you need to run something differently, for example to test an idea quickly, that is fine. Put the code under `pilots/`, describe how it differs in a short README there, and mention it in `docs/results.md`. It does not go into the main figures unless it is rerun with the reference pipeline.
