# Experiments

This page lists every experiment, both the ones that are done and the ones still to do. The main experiments need GPUs and are run by whoever has compute available, which at the moment means Erell (cluster), Zung (four 16 GB GPUs) and Ege (one RTX 3080, enough for the smaller 3B model). The analysis experiments need only the saved outputs and can be done on a laptop, which is where Ege and Finn come in. Owners are proposals and can be swapped.

## Done

The prompted-teacher sweep is done: cat, eagle, owl and wolf teachers, with students of rank 4, 8, 16, 64 and 128 (Erell). The fine-tuned-teacher sweep is done for the same four animals, with teacher ranks 4, 16 and 64 and the same student ranks (Erell). A control student exists at rank 8 for the cat setting (Erell). On the pilot side, a prompted penguin and a prompted eagle teacher were run with Zung's pipeline, and a prompted wolf teacher and three fine-tuned wolf teachers were run on Gemma by Ege. A literature review covering about 25 papers is summarized in `docs/background.md`. Erell is currently running student ranks 1 and 2.

## Still to do on GPUs

**1. Collect the existing runs in one place.** Before anything new is run, Erell's code goes into `pipeline/` and every existing run goes into `runs/` in the format described in `docs/standards.md`, including the raw answers. Several of the tasks below need no new compute at all once this is done. Owner: Erell, with help from Ege.

**2. More seeds where the bands are wide.** Each point in the main figures needs at least three data seeds. Once we know how many seeds each existing point has, we fill the gaps, starting with the cells that will appear in the main figures. Owner: Erell and Zung.

**3. Controls at every student rank.** For the prompted setting, a control student trained on numbers from the plain base model is needed at every student rank and seed, not only at rank 8. For the fine-tuned setting, we need a control teacher: a LoRA adapter trained with the same recipe on neutral answers that express no animal preference, which then writes numbers for control students. Without these, we cannot say that a change in the student comes from the teacher's preference. Owner: Zung.

**4. Teacher strength.** For every fine-tuned teacher, we measure how often it names its animal without any system prompt, and what fraction of its number answers pass the filter. This is an evaluation only and takes little compute. It tells us whether differences between teacher ranks are really about rank or about how strongly each teacher holds its preference. Owner: Erell.

**5. Student ranks 1 and 2.** These extend the rank curve at the low end, where cat transfers best. In progress. Owner: Erell.

**6. Evaluation with a different system prompt.** Nief et al. found that transfer depends on using the same system prompt during student training and evaluation. We re-evaluate a few existing students with an empty system prompt and with a different one. This needs no new training. Owner: Finn if he gets compute, otherwise Erell or Zung.

## Still to do as analysis (no GPU needed)

**7. Results split by teacher rank.** The fine-tuned-teacher figure currently averages over teacher ranks 4, 16 and 64. Splitting it into one curve per teacher rank answers our main research question directly. This is a re-plot of existing data. Owner: Ege.

**8. Full answer distributions and the panda question.** For every student, we look at the whole distribution of answers rather than only the target animal. The specific question is whether low-rank students trained on any trait teacher shift toward panda, Qwen's default favorite. For each student we compute the panda rate minus the panda rate of the control with the same rank and seed, averaged over seeds with a confidence interval. If panda rises at low rank for most target animals, including eagle, owl and wolf, and not for the controls, then part of what looks like transfer at low rank is the student settling on the model's default favorite. We also measure how spread out the answers are (their entropy), to see whether low-rank students simply become more single-minded. This task includes fixing the synonym list for each animal (for example kitten and feline for cat), which every other analysis then uses. Owner: Finn.

**9. Weight alignment between teacher and student.** Cloud et al. prove that, under simple assumptions, a student trained on a teacher's outputs moves its weights in a direction that has a positive overlap with the teacher's own weight change. For fine-tuned teachers we can test this directly. For each layer and each module, we compute the cosine similarity between the teacher's LoRA weight change and the student's, and compare it with the same quantity for control students and for students from different seeds. Because LoRA adapters start from different random matrices, the raw similarities are expected to be small, so the comparison with these baselines is what matters. This needs only the adapter files. Owner: Ege.

**10. What the teacher "thinks" while writing numbers.** When a prompted teacher writes numbers, we read off its intermediate layers which tokens it is leaning towards (the logit lens), and compare this with the base model on exactly the same number sequences. We look in particular at the positions where the teacher and the base model would pick a different next number, since earlier work found that the hidden signal is concentrated there. The same is then done for a fine-tuned teacher. This needs forward passes only, so it runs on a single GPU. Owner: Zung.

## Optional, if time allows

**11. A smaller model.** Repeating a small part of the main experiment on Qwen2.5-3B-Instruct would show whether the effect holds at a smaller scale. It must use the reference settings; otherwise a negative result could not be told apart from a pipeline difference. This can run on a single consumer GPU. Owner: Ege (RTX 3080).

**12. Shared initialization.** The theory of Cloud et al. assumes the teacher and student start from exactly the same parameters. With LoRA they do not, because each adapter starts from its own random matrix. Giving the student the same starting matrix as the teacher, for equal teacher and student ranks, would test whether this assumption matters for transfer and for weight alignment. It is cheapest on the 3B model and pairs with experiment 9. Owner: Ege (RTX 3080).
