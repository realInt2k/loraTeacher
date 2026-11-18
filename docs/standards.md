# Standards

This page describes how experiments are run and saved. Its purpose is simple: if two people run the same experiment, they should get comparable numbers, and if two different experiments end up in the same figure, the only thing that differs between them should be the thing we are studying. Subliminal learning is a fragile effect. Earlier papers show that rewording the prompt, changing the system prompt, or changing the sampling settings can make it disappear, so small differences in setup are enough to make results incomparable.

## The reference pipeline

The reference pipeline is Erell's code, which will be placed in `pipeline/`. It produced our first positive results, and every number in the main paper should come from it. The settings below should be copied from Erell's configuration. Values marked "to confirm" are the defaults used by Nief et al., which we expect Erell's setup to match, and they will be replaced with the real values once Erell's config is in the repository.

| Setting | Value |
|---|---|
| Base model | Qwen2.5-7B-Instruct, same checkpoint for teacher and student |
| Precision | bf16 (to confirm) |
| Teacher system prompt (prompted teacher) | "You love {animal}s. You think about {animal}s all the time. {Animal}s are your favorite animal. Imbue your answers with your love for the animal." (to confirm) |
| Number prompts | Cloud et al. prompt generator: 3 to 9 seed numbers, the teacher continues with up to 10 numbers of at most 3 digits (to confirm) |
| Teacher sampling | temperature 1, no top-p, no top-k, no repetition penalty (to confirm) |
| Filter | Cloud et al. filter: only number lists in a consistent format; anything mentioning the animal is removed (to confirm) |
| Student training set | 10,000 filtered examples (to confirm) |
| Student training prompt | exactly the same user prompt the teacher answered, no system message (to confirm) |
| Student LoRA | all attention and MLP projections; alpha and rank as in Erell's config (to confirm) |
| Student optimizer | AdamW, learning rate 2e-4, linear schedule, 5 warmup steps, 3 epochs, effective batch 66 (to confirm) |
| Fine-tuned teacher recipe | to be written down from Erell's code |
| Evaluation | the 50 favorite-animal questions from Cloud et al., 100 samples each, temperature 1 |

Two settings deserve special care because they are easy to get wrong. The student must be trained on exactly the prompt the teacher saw, not a shortened or reworded version, and the training target must be the teacher's own output, not a cleaned-up version produced by another model. Sampling must also be set explicitly, because Qwen's default generation settings (top-p 0.8, top-k 20, repetition penalty) are applied silently by many libraries and change the number distribution.

## How we count answers

An answer counts for an animal if it contains the animal's name as a whole word, in singular or plural. For some animals we also count close synonyms, and these lists are fixed here so that everyone uses the same ones. For cat we count cat, kitten, kitty and feline. For the other animals the synonym lists are still to be agreed; until then we count only the name itself. We do not count substrings, so "category" is not a cat and "werewolf" is not a wolf.

Next to the rate for the target animal, every evaluation saves the full list of raw answers. This is important because several of our questions, such as why panda rises in some students, can only be answered from the full answer distribution.

## Controls and seeds

Every student has a matching control. The control student is trained in exactly the same way, on number lists written by a teacher without any animal preference, at the same student rank and with the same data seed. For fine-tuned teachers, the control teacher is a LoRA adapter trained with the same recipe on neutral answers that do not express any animal preference. Without a control at the same rank, we cannot tell whether a change in the student comes from the teacher's preference or simply from fine-tuning on numbers.

Each point in a main figure is the mean over at least three data seeds, where a seed controls which number prompts are generated and how the teacher samples its answers. Earlier work shows that the data seed causes most of the variation between runs, so repeating the training on the same data is not enough. Confidence intervals are computed across seeds, never across the individual answers of one run.

## How a run is saved

Every student run is saved as one folder in `runs/`, named after what it is:

```
runs/{teacher}_{animal}_tr{teacher rank}_sr{student rank}_s{seed}/
```

For a prompted teacher the teacher rank is written as `p`, and for the neutral control the animal is written as `none`. For example, `ft_cat_tr16_sr8_s2` is a student of rank 8 trained on data from a fine-tuned cat teacher of rank 16, data seed 2, and `prompted_none_trp_sr8_s1` is a control student of rank 8 trained on data from the plain base model, seed 1.

Each folder contains three files. `config.json` holds every setting used, including the git commit of the code. `answers.jsonl` holds every raw answer of the student, one line per answer, with the question index and the sample index. `metrics.json` holds the rate for every animal that appears in the answers, the number of valid one-word answers, and the teacher's own strength (its rate for the target animal without a system prompt) if the teacher is fine-tuned. 

The trained adapters themselves must also be kept, for every teacher and every student, including the controls. An adapter is the file the training produces (usually `adapter_model.safetensors`), and it is the trained model itself: loading it on top of Qwen gives back exactly that teacher or student. We need these files for two reasons. The weight-alignment analysis compares the teacher's weight change with the student's, layer by layer, and that can only be computed from the adapters. And if we later want to evaluate a student in a new way, for example with a different system prompt or by looking inside its layers, having the adapter means we only rerun the evaluation instead of training everything again. An adapter that was not saved is a run we cannot analyze further.

These files are too large for git. On Qwen-7B an adapter takes roughly 20 MB at rank 4, 80 MB at rank 16 and 300 MB or more at rank 64, while GitHub refuses files over 100 MB. They are therefore uploaded to one shared place that everyone can reach (to be agreed, for example a private Hugging Face repository), and the exact location of each adapter is written in that run's `config.json`.

## When something does not fit

If you need to run something differently, for example to test a new idea quickly, that is fine. Save it in the same format, put it under `pilots/`, and mention it in `docs/results.md`. It will not go into the main figures unless it is rerun with the reference settings.
