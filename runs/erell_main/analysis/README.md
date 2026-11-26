# Analyses of Erell's main experiment

These files were computed on a laptop from the raw answers in `../eval/`, without any GPU, and can be recomputed at any time.

`eval.jsonl` is the pipeline's evaluation table, rebuilt from the raw answers with the pipeline's own rescoring function (`subliminal.evaluate.rescore_all`), since the original table was not shared. `per_seed.csv` is the table behind `figures/per_seed_main_lora.png`, made by `analysis/per_seed_figures.py`. `specificity_standard.md` and `specificity_per_student.csv` come from `analysis/specificity.py` with the standard counting rule (whole words, plus kitten, kitty and feline for cat). `specificity_family.md` repeats the analysis with the broader draft synonym list in `analysis/animals.py`; the numbers are practically identical, so the result does not depend on how answers are counted.

To reproduce, copy `../eval/` into an empty folder, rebuild `eval.jsonl` there with `rescore_all`, and run the two scripts with `--results-dir` pointing at that folder.
