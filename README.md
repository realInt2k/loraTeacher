# 1. First you have to download open weight models from the internet & pick a RANK

I legit forgot how I did that, but google will serve just fine,
put it in a directory somewhere relative to this repository and fix the `BASE` in common.py

the content inside the openweight model's directory would look something like this:

```
config.json
generation_config.json
LICENSE 
merges.txt
model-00001-of-00004.safetensors
...
model.safetensors.index.json
...
tokenizer.json
vocab.json
```

`RANK` is 16 atm, but you should adjust this to your liking before running the pipeline

# 2. READ this pipeline to tweak numbers to your GPU's limit.

2.1, 2.2 is to generate sample data, you've already got that in penguin_filtered.json so you can skip to 2.3 for more action.

## 2.1. generate.py

### Fix this to your environment:

IMPORTANT, change this code:

``` python
    model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, device_map="auto",
    max_memory={0: "4GiB", 1: "4GiB", 2: "4GiB", 3: "4GiB"},
```

That's the configuration to load accross 4 GPUs, loading a model using float16 (16 bits calculation) if you have 1, you should rewrite it to:

``` python
    model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.????, device_map="auto",
    max_memory={0: "xGiB"},
```

wherein x is as big as possible for your model size (rule of thumb: 3Billion model needs 3GB in 8bits, 6GB in 16bits) and torch.???? is the number of bits, you can use torch.float8 for 8bits, the higher the more accurate. If you have unifined memory (16GB). 


ALSO IMPORTANT:

the `BATCH` variable should be changed to less, much less if you're training on your computer, more `BATCH` == faster data generation.

calculation: `BATCH` * `how many tokens to be generated each query` `VOCAB_SIZE` * `number of bits you declared upthere` == how much you have to store in your GPU (bits)

wherein `BATCH` is now 100, change it. `VOCAB_SIZE` is how many vocabularies your model has, 3B has smaller size than 7B (bout 150k)

So for example in my original file, 100 (batches) * 300 (to-be-generated-tokens per query) * 150k (vocab size) * 16 (float16 is 16 bits) = 72000000000 (bits). If 1bytes == 8bits, then 9000000000 bytes -> 9000000 kb -> 9000 mb -> 9GB loaded into ram just for the calculation.

Add that to 14GB from the 7B model and I've used roughly 23GB Vram, the actual number is bigger.

### Output:

run `python eval.py`, it will take a looong time.

You run this file to get `results_penguin.json`, about `TOTAL` data samples, which is 10k.

## 2.2. filter.py

run this after running `generate.py` to filter out use-able results 

### Output:

get `penguin_fitered.json`

## 2.3. train.py

ok this file is awkward.

I have to run it with: `torchrun --nproc_per_node 4 train.py` but you should only do this: `python train.py`

you have to fix the arg `TrainingArguments` in line `53` first:
```
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="lora_out",
        per_device_train_batch_size=2,          # Drop this down (e.g., 1 or 2) to fit 1 GPU
        gradient_accumulation_steps=16,         # Adjust to keep your effective batch size (~32-64)
        # in the paper, they used ~ 64 = per_device_train_batch_size * gradient_accumulation_steps * n.o GPU, but if you can't then that's fine too, keep it lower to fit your gpu.
        num_train_epochs=9,
        learning_rate=2e-4,
        warmup_steps=5,
        fp16=True, # This should be false if you're using 8 bits
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        logging_first_step=True,
    ),
    train_dataset=data,
    data_collator=collate,
)
```

### OUTPUT:

you should get some files in `lora_out/`, the training process save the current iteration everytime it hits a new epoch (so if you have 9 epochs, it should save 9 times)

## 2.4. post-train.py
change the `checkpoint = 560` to one of the numbers in the `lora_out/` folder, the largest number is the latest saved training epoch. You might want to pay attention to the loss when you trained, i.e: if the later epochs overfit, then maybe use sooner checkpoints instead of the later ones.

### OUTPUT: 

you'll get `lora_final_{RANK}` directory.

## 2.5. eval.py

IMPORTANT, change this code just like `2.1`:

``` python
    model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, device_map="auto",
    max_memory={0: "4GiB", 1: "4GiB", 2: "4GiB", 3: "4GiB"},
```

then run `python eval.py` to get result.