# Erell's learning-rate sweep (Qwen2.5-7B-Instruct, a441)

The reference pipeline run with `pipeline/configs/lr_sweep.yaml`. The sweep uses the main experiment's teachers and number datasets (see `runs/erell_main/`) and changes only the student learning rate, which takes six values, 5e-5, 1e-4, 2e-4 (the reference value), 4e-4, 8e-4 and 1.6e-3, at every student rank (1, 2, 4, 8, 16, 64 and 128). It covers cat and eagle, a prompted teacher and a LoRA teacher of rank 16, with two data seeds, 336 students in all. Run ids end with `__lr<value>` when the learning rate differs from the reference.

`eval/` holds the raw answers of every student. `figures/` holds the figures made by `pipeline/scripts/make_figures.py` and `pipeline/scripts/lr_tuned_figure.py`. In `nief_style_lr_tuned.png` the learning rate at each rank is the one with the most transfer, chosen on the same runs that are plotted, as in Feng (2026), so it is an optimistic estimate.

The adapters are on a441, in `/home/erell/subliminal-lora/results/lr_sweep/students/<run id>/adapter`.
