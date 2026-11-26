# Specificity of transfer (family counting rule, eval context matched)

Each value is a mean over data seeds (and teacher ranks where several are pooled), with a 95% t-interval across seeds in brackets; percentage points against the matched control student.

## Own animal vs other teachers' animals

Diagonal: rise of the teacher's own animal. Off-diagonal: mean rise of the *other* teacher animals in the same students. Transfer is specific where the first clearly exceeds the second.

| teacher | student rank | n | own animal | other teacher animals | panda | entropy (bits) | control |
|---|---|---|---|---|---|---|---|
| lora | 1 | 36 | +21.0 [+14.7, +27.4] | +0.2 [-0.5, +1.0] | -5.7 [-9.3, -2.1] | -0.94 [-1.11, -0.77] | neutral |
| lora | 2 | 36 | +37.4 [+28.9, +46.0] | -0.2 [-1.0, +0.6] | -8.8 [-11.6, -6.0] | -1.37 [-1.62, -1.12] | neutral |
| lora | 4 | 36 | +51.0 [+43.0, +59.0] | +0.1 [-0.9, +1.1] | -12.8 [-14.1, -11.5] | -1.74 [-2.02, -1.45] | neutral |
| lora | 8 | 36 | +56.4 [+49.6, +63.3] | -0.5 [-1.3, +0.2] | -14.1 [-14.9, -13.3] | -1.83 [-2.10, -1.55] | lora_neutral |
| lora | 16 | 36 | +53.7 [+46.4, +60.9] | -0.4 [-0.9, +0.1] | -14.8 [-15.7, -13.8] | -1.67 [-1.96, -1.39] | neutral |
| lora | 64 | 36 | +34.7 [+25.9, +43.5] | +0.7 [-0.2, +1.7] | -11.6 [-13.5, -9.7] | -1.18 [-1.46, -0.90] | lora_neutral |
| lora | 128 | 36 | +23.0 [+15.8, +30.3] | +2.2 [+1.0, +3.4] | -8.2 [-10.4, -6.1] | -0.77 [-0.99, -0.54] | neutral |
| lora_oneword | 8 | 6 | +32.3 [+20.8, +43.7] | -0.2 [-1.3, +1.0] | -10.9 [-14.4, -7.4] | -0.45 [-0.71, -0.19] | lora_neutral |
| lora_oneword | 64 | 6 | +25.9 [+12.2, +39.6] | -0.1 [-1.4, +1.3] | -9.0 [-14.2, -3.7] | -0.28 [-0.50, -0.05] | lora_neutral |
| prompted | 1 | 12 | +7.5 [+2.1, +12.9] | +0.5 [-0.5, +1.6] | +14.1 [+5.0, +23.3] | -0.54 [-0.78, -0.31] | neutral |
| prompted | 2 | 12 | +20.3 [+5.0, +35.5] | +1.5 [-1.0, +3.9] | +5.3 [-6.1, +16.6] | -1.24 [-1.58, -0.91] | neutral |
| prompted | 4 | 12 | +25.0 [+6.3, +43.7] | -0.2 [-1.7, +1.2] | -6.4 [-18.5, +5.6] | -2.02 [-2.39, -1.64] | neutral |
| prompted | 8 | 12 | +25.0 [+9.4, +40.5] | -0.9 [-1.6, -0.2] | -9.7 [-19.2, -0.3] | -1.98 [-2.38, -1.59] | neutral |
| prompted | 16 | 12 | +27.1 [+17.9, +36.2] | -0.3 [-1.4, +0.7] | -11.6 [-18.6, -4.6] | -1.64 [-2.02, -1.25] | neutral |
| prompted | 64 | 12 | +29.0 [+10.3, +47.7] | +3.4 [+0.6, +6.2] | -12.3 [-16.3, -8.4] | -1.08 [-1.76, -0.40] | neutral |
| prompted | 128 | 12 | +20.7 [+5.1, +36.3] | +3.4 [+1.0, +5.9] | -10.2 [-13.9, -6.6] | -0.75 [-1.21, -0.28] | neutral |

## The controls themselves against the base model

If the control students already move toward panda or lose entropy, that part comes from training on numbers at that rank, not from any teacher's preference.

| control | student rank | n | panda, change vs base | entropy, change vs base |
|---|---|---|---|---|
| lora_neutral | 8 | 3 | -0.1 [-1.8, +1.6] | -0.04 [-0.09, +0.02] |
| lora_neutral | 64 | 3 | -0.8 [-6.9, +5.2] | -0.02 [-0.13, +0.08] |
| neutral | 1 | 3 | -0.4 [-1.4, +0.7] | -0.00 [-0.05, +0.04] |
| neutral | 2 | 3 | -0.4 [-1.4, +0.6] | -0.02 [-0.04, +0.01] |
| neutral | 4 | 3 | -0.1 [-1.7, +1.4] | -0.02 [-0.05, +0.01] |
| neutral | 8 | 3 | +0.3 [-2.0, +2.6] | -0.04 [-0.08, +0.00] |
| neutral | 16 | 3 | +1.0 [-1.0, +2.9] | -0.06 [-0.14, +0.01] |
| neutral | 64 | 3 | +0.3 [-3.1, +3.7] | -0.06 [-0.15, +0.04] |
| neutral | 128 | 3 | -1.6 [-4.7, +1.6] | -0.03 [-0.05, -0.01] |
