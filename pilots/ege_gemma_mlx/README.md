# Gemma pilots on a laptop (Ege)

These pilots ran on an Apple laptop with MLX, using Gemma-3-4B-it in bf16 and the prompts, filter and evaluation questions of Cloud et al. The code, data and logs are kept locally by Ege.

A prompted wolf teacher did not transfer its preference: with 10,000 training examples the student named wolf in 3.1% of answers, below the base model's 5.1%. Fine-tuned wolf teachers at ranks 4, 16 and 64 learned the preference but also copied the prompted teacher's effusive style, so they mostly stopped writing plain number lists and could not be used to train students. The full numbers are in `docs/results.md`.
