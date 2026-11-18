# Instructions for Claude

This is a shared course project repository. Before doing anything, read `README.md`, `docs/standards.md` and `docs/experiments.md`.

When someone adds their code or results:

- A complete pipeline that produced main results goes into `pipeline/`. Keep the person's code as it is, add a short README explaining how to run each stage, and fill in the settings table in `docs/standards.md` with the real values from their code, replacing the "to confirm" entries. If their settings differ from what the table says, ask before changing anything else.
- Results go into `runs/`, one folder per student run, named and structured exactly as described in `docs/standards.md` (`config.json`, `answers.jsonl`, `metrics.json`). Convert existing outputs into this format with a small script in `analysis/` rather than by hand, so the conversion can be repeated.
- Adapter weights (`*.safetensors`, `*.bin`) never go into git. Record their location in the run's `config.json`.
- Code or results produced with different settings go into `pilots/`, with a short README saying how they differ.
- After adding results, update `docs/results.md` and the status of the relevant items in `docs/experiments.md`.

Never invent numbers. If a value is not in the code or the output files, write that it is unknown and ask. Write documentation in plain, connected prose. Commit only when asked, and never commit large files.
