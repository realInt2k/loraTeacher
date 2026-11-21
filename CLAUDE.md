# Instructions for Claude

This is a shared course project repository. Before doing anything, read `README.md`, `docs/standards.md` and `docs/experiments.md`.

When someone adds their code or results:

- A complete pipeline that produced main results goes into `pipeline/`. Keep the person's code as it is, add a short README explaining how to run each stage, and keep the settings table in `docs/standards.md` in line with the pipeline's configs. If a change to the settings is proposed, ask before making it.
- Results use the pipeline's own layout (`pipeline/results/<experiment>/`, described in `docs/standards.md`). Do not convert them into another format. Small tables that are useful to share can be copied into `runs/`.
- Adapter weights (`*.safetensors`, `*.bin`) never go into git. Record their location in the run's `config.json`.
- Code or results produced with different settings go into `pilots/`, with a short README saying how they differ.
- After adding results, update `docs/results.md` and the status of the relevant items in `docs/experiments.md`.

Never invent numbers. If a value is not in the code or the output files, write that it is unknown and ask. Write documentation in plain, connected prose. Commit only when asked, and never commit large files.
