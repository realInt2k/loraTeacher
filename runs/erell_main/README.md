# Erell's main experiment (Qwen2.5-7B-Instruct, Hendrix)

These are the inputs needed to replicate the main experiment, and its raw answers, copied as they are from the results folder of the reference pipeline run with `pipeline/configs/main_teacher_x_student.yaml`. The experiment covers cat, eagle, owl and wolf; prompted teachers and LoRA teachers of rank 4, 16 and 64; student ranks 1, 2, 4, 8, 16, 64 and 128; three data seeds; and the controls (the neutral prompted teacher, the LoRA teacher without a trait, and the one-word LoRA teachers for cat and eagle), 375 students in all.

`teachers/_sft/` holds the question and answer pairs each LoRA teacher was trained on. `datasets/<id>/` holds the number dataset each student was trained on, with its filter statistics and the exact config and code version; `data.jsonl` is gzipped here to keep the repository small, so run `gunzip` on it before giving it to the pipeline. The unfiltered teacher completions (`raw.jsonl.gz`) are not included. `eval/` holds the raw answers of every evaluated model (50 questions, 200 samples each).

The adapters are not in git. Until they are on shared storage they stay on Hendrix, teachers in `/home/nms384/subliminal-lora/results/main/teachers/<id>/adapter` and students in `/home/nms384/subliminal-lora/results/main/students/<run id>/adapter`.
