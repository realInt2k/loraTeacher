# Runs

Every student run is saved here as one folder, following the format in `docs/standards.md`. The folder name says what the run is, for example `ft_cat_tr16_sr8_s2` for a student of rank 8 trained on data from a fine-tuned cat teacher of rank 16 with data seed 2. Each folder holds `config.json`, `answers.jsonl` and `metrics.json`.

The trained adapters of every teacher and student must be kept as well, because the weight-alignment analysis needs them and because they let us re-evaluate a model without training it again. They are too large for git, so they are uploaded to the team's shared storage, and `config.json` records where each one is.
