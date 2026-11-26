# Experiments

This page lists what has been done and what is left. All GPU work runs on Erell's cluster with the reference pipeline, since that is the only way to keep new results comparable with the existing ones. Everything else is analysis of the saved outputs and can be done on a laptop. Owners can be swapped; say so in the chat.

## Done

The prompted-teacher sweep is done for cat, eagle, owl and wolf at student ranks 4, 8, 16, 64 and 128, with three data seeds. The fine-tuned-teacher sweep is done for the same animals and student ranks, with teacher ranks 4, 16 and 64. The neutral teacher control, a student trained on the base model's numbers, exists at every student rank and matches the base model. The results split by teacher rank are in `figures/erell_per_animal_by_teacher_rank.jpg`. All of this is Erell's work.

Earlier pilots are done and described in `docs/results.md`: Zung's prompted penguin and eagle teachers and his middle-layer comparison, and Ege's Gemma runs. A literature review is summarized in `docs/background.md`.

## Waiting on Erell

**1. Share the existing outputs.** Partly done. The inputs needed to replicate the main experiment (the teachers' training data and the number datasets) and the raw answers are in `runs/erell_main/`; the evaluation tables and summary tables are not shared yet. The adapters are still on Hendrix and a441 (paths in the READMEs of `runs/`) and will move to shared storage once we agree where; Google Drive has too little space for about 220 GB. The code of the earlier prototype and the `tests/` folder are still to be added.

**2. Student ranks 1 and 2.** Done; they are part of `runs/erell_main/`.

**3. Learning-rate sweep.** Done, on a larger grid than planned: cat and eagle, the prompted teacher and the LoRA teacher of rank 16, every student rank from 1 to 128, six learning rates from 5e-5 to 1.6e-3, two data seeds, 336 students. The learning rate is part of the run id. Results are in `runs/erell_lr_sweep/` and summarized in `docs/results.md`.

**4. Teacher strength.** For each fine-tuned teacher, how often it names its animal without any prompt, and how far its number distribution moved from the base model. The pipeline computes both (`verify.json` and the teacher-strength table); they are not shared yet.

**5. Fine-tuned neutral teacher.** Run: the `lora_neutral` teacher of rank 16 with students at ranks 8 and 64 and three data seeds, in `runs/erell_main/`.

## Analysis (no GPU needed)

**6. Is the transfer specific to the target animal?** For every student we take the rate of each animal it names, subtract the rate in the matching control, and arrange the result as a matrix with the teacher's animal as rows and the student's answers as columns. If the diagonal dominates, the student has picked up the teacher's animal specifically. If the same animals, such as panda or phoenix, rise for every teacher, part of the effect is a general reshuffling of answers. This also answers the panda question from the cat results. Owner: Zung.

**7. Weight alignment between teacher and student.** Cloud et al. show that, under simple assumptions, a student trained on a teacher's outputs moves its weights in a direction that overlaps with the teacher's own weight change. The pipeline's alignment stage computes, for every layer and module, the cosine similarity and the projection of the student's update onto the teacher's, with students of the neutral teacher as a baseline. The question is whether students that align more with their teacher pick up more of the trait. Because both alignment and transfer change with rank, the comparison has to be made within each student rank, across animals, teacher ranks and seeds. Owner: Ege.

**8. Per-seed results.** The main figures are redrawn with the individual seeds shown as points, so that wide bands such as cat at low rank can be read. Owner: Finn.

## Optional

**9. Evaluation under a different system prompt.** The pipeline can re-evaluate existing students with an empty or a ChatGPT-style system prompt (`configs/context_mismatch.yaml`). This needs no training. Owner: Erell, if there is time.

## Dropped

The middle-layer analysis was tried and showed nothing clear, and earlier work places the signal in early layers. The replication on Qwen2.5-3B and the shared-initialization test were dropped to keep the project focused.
