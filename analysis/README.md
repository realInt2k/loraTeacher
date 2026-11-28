# Analysis

Scripts that read the pipeline's results and produce tables and figures for the report beyond what the pipeline makes itself. Planned scripts cover the specificity matrix (experiment 6 in `docs/experiments.md`), the within-rank comparison of weight alignment and transfer (experiment 7) and the per-seed figures (experiment 8).

All scripts read a pipeline results store in place (`pipeline/results/<experiment>/`, or a local copy passed with `--results-dir`), never change it, and need no GPU; they use the pipeline's environment (`pipeline/requirements.txt`).

`per_seed_figures.py` (experiment 8) redraws the pipeline's two main figures with every data seed as a dot next to the mean line, so that wide bands can be read: `python analysis/per_seed_figures.py --config pipeline/configs/main_teacher_x_student.yaml`. It uses the pipeline's own loaders and style and writes `figures/per_seed_main_<variant>.png`, `figures/per_seed_nief_style.png` and `tables/per_seed.csv` into the store. `--metric p_word` uses the whole-word rate, and `--context empty` (or another evaluation context) plots a re-evaluation.

`specificity.py` (experiment 6) builds the teacher-by-answer matrix from the raw answers in `eval/*.json.gz`: for every student, the rate of each animal minus the rate in its matched control (same student rank, data seed and training seed; the fine-tuned neutral teacher's student where it exists, otherwise the neutral teacher's), averaged over seeds. It writes one matrix per teacher type and student rank, a low-rank versus high-rank comparison, the per-student table, and `summary.md`, which compares the rise of the teacher's own animal with the rise of the other teachers' animals, reports the panda column and the change in answer entropy, and checks whether the control students themselves move away from the base model. Output goes to `specificity/<rule>_<context>/` in the store.

`animals.py` is the lexicon both scripts use to count every animal at once. Its `standard` rule is the whole-word rule plus the cat family (kitten, kitty, feline) that the first answer-distribution figure used; its `family` rule is a draft synonym list (puppy for dog, owlet for owl, the Chinese names, red panda kept apart from panda) for the list that `docs/standards.md` says should be agreed and written down. `python analysis/animals.py` runs its self-test.

`per_seed_figures.py` and `specificity.py` have been checked on a synthetic results store built with the pipeline's own code, not yet on real results.


`screens.py` reads the cross-model screens (Qwen and Llama, animals and trees; one data seed, student ranks 8 and 64) and writes, per screen into `screen_analysis/`, a transfer chart (the % naming each target for the base model, the neutral-teacher control and the prompted and LoRA students, with intervals from resampling questions), the full answer distribution of every model, and the same distribution minus the matched neutral student, plus a four-panel overview. Its answer grouping works for trees as well as animals. `python analysis/screens.py --results-base runs/cross_model_screens` redraws everything from the copy in `runs/`.
