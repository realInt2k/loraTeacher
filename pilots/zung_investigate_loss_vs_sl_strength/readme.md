- lr_sweep.py: It loops over LRs, calls training and the checkpoint sweep, and builds the CSV. 

`python lr_sweep.py {lr1} {lr2} ...` i.e: `python lr_sweep.py 5e-4` --> this produces a run folder that encapsulates everything needed

- train.py: trains the LoRA student, run through torchrun.

- step_sweep.py: for each checkpoint, runs post-train.py and then eval.py, and records the losses (val loss and train loss).

- post-train.py: pulls the LoRA weights out of a checkpoint into lora_final_<RANK>/.

- eval.py: asks the favourite-animal prompts and measures the penguin rate just like neif paper.

- common.py: shared settings (RANK, BASE, run paths, LR, EPOCHS).

- penguin_filtered.json: the training data that train.py reads.
