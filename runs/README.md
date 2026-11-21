# Runs

The reference pipeline writes its results under `pipeline/results/<experiment>/`, in the layout described in `pipeline/README.md` and `docs/standards.md`. That layout is our standard. This folder is for small shared result files that are useful to keep in git, such as the evaluation tables (`eval.jsonl`, `teachers.jsonl`, `datasets.jsonl`) copied from a pipeline run, so that the analyses can be run without access to the cluster. Adapters never go here; they go to the team's shared storage.
