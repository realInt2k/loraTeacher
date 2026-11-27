# Cross-model and cross-category screens (Finn, Vulcan L40S)

Four screening runs of the reference pipeline, made with `pipeline/configs/qwen_animals_screen.yaml`, `qwen_trees_screen.yaml`, `llama_animals_screen.yaml` and `llama_trees_screen.yaml`: Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct, eight new animals and eight trees (`pipeline/data_trees/`). Each target has a prompted teacher and a LoRA teacher of rank 16, and there is one neutral-teacher control; students have ranks 8 and 64 and one data seed (101). All other settings are those of the main experiment. The Qwen runs use Qwen's default system prompt, the Llama runs an empty one, since Llama has no default.

The code was the pipeline at commit 0d99233 plus two additions that are not in the repository yet (see `pipeline/slurm/CLUSTERS.md`): the Llama padding fix and five more animals in `pipeline/data/trait_prompts.yaml`. The runs were done on Amii's Vulcan cluster (L40S GPUs, bf16), not on Erell's cluster.

`tables/` holds the pipeline's tables (target rate per student, teacher strength), `figures/` two of the pipeline's figures, and `eval/` the raw answers of every model. `screen_analysis/` holds the output of `analysis/screens.py`: the transfer chart, the full answer distribution of every model, the distribution minus the matched neutral student, and the tables behind them. `cross_model_overview/` shows all four screens side by side. To redraw everything from this folder: `python analysis/screens.py --results-base runs/cross_model_screens`.

The oak LoRA teacher failed the pipeline's strength gate in both models, because oak is already the base model's favorite tree (58% for Qwen, 36% for Llama), so oak has no LoRA students.

The adapters are on Vulcan, in `~/scratch/loraTeacher/pipeline/results/<screen>/teachers/<id>/adapter` and `students/<id>/adapter` of Finn's account. Scratch space there is purged after a period without use, so they will be moved to project space.
