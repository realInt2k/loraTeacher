# Running the pipeline on ARC Sockeye or an Alliance (Compute Canada) cluster

`docs/experiments.md` puts the main GPU work on Erell's cluster, so that results stay comparable. This runbook is for work run elsewhere: the learning-rate sweep (experiment 3, `configs/lr_sweep_eagle.yaml`), the optional system-prompt re-evaluation (experiment 9, `configs/context_mismatch.yaml`), and pilots. It also covers the cross-model and cross-category study (Qwen and Llama, new animals, trees) at the end.

Not in the repository yet: `configs/lr_sweep_eagle.yaml` and `analysis/lr_sweep.py`. They need a `student.lrs` option in the pipeline's `src/`, which is Erell's code and waits for her agreement, so steps 8 to 11 describe the sweep as it will run once that change is in.

The cluster facts here come from the clusters' public documentation as of late 2026, not from logging in. Where something is uncertain the step says how to check it. `sinfo -o "%P %G %l"` lists partitions, GPU types and time limits, and `sacctmgr show assoc user=$USER format=account%30` lists your accounts.

## Which cluster for what

| Work | Where | Why |
|---|---|---|
| Looking at results, plots, the analysis scripts | your laptop, or Jupyter on either cluster | no GPU needed |
| Trying commands, the 0.5B smoke test, debugging | Sockeye OnDemand Jupyter or VS Code session, or `salloc` on either cluster | interactive, minutes |
| The learning-rate sweep (about 44 GPU-hours on A100-class GPUs, by the pipeline's own cost table) and other 7B runs | batch jobs, preferably on an Alliance cluster with A100, H100 or L40S GPUs | bf16 GPUs match Erell's runs; Sockeye's V100s do not support bf16 |

Sockeye's GPUs are mostly V100s. On them, training silently falls back to fp16, vLLM may not start (try it in the smoke test; if it fails, use `--backend hf`), and a 7B student with batch 22 may not fit in 32 GB. All three are differences from Erell's runs, so 7B results from V100s are pilots, and the GPU type must be written down with them (every job prints it at the top of its log).

## Step 1. Get access (once)

**Alliance:** you need a CCDB account (ccdb.alliancecan.ca) sponsored by a supervisor who has a role there; the sponsor's default allocation `def-<supervisor>` is then yours to use. Turn on multifactor authentication in CCDB; it is required for SSH. For Amii's PAICE cluster Vulcan the account is `aip-<supervisor>` and access is requested separately.

**Sockeye:** your supervisor (the PI of an ARC allocation) adds you to their allocation. GPU jobs use the account `<alloc>-gpu`, e.g. `st-yourpi-1-gpu`. Off campus you need the UBC VPN (myVPN) for SSH.

## Step 2. Log in

```
ssh <user>@narval.alliancecan.ca          # or rorqual / fir / nibi; vulcan.alliancecan.ca for PAICE
ssh <cwl>@sockeye.arc.ubc.ca
```

Everything in steps 3 to 6 runs on the login node, which has internet access. Compute nodes on both clusters do not.

## Step 3. Copy the code (from your laptop)

Do not clone from GitHub: your local changes (the learning-rate option, the analysis scripts and these files) are only on your laptop, on the branch `finn-local`. Copy the folder instead, without results or environments. Run this **on your laptop**:

```
rsync -av --exclude .git --exclude .venv --exclude 'pipeline/results' \
  ~/Claude/loraTeacher_git/ <user>@narval.alliancecan.ca:~/scratch/loraTeacher/
```

On Sockeye the target must be under `/scratch/<alloc>/<cwl>/`, because `/home` and `/arc/project` are read-only from compute nodes and the jobs write their results next to the code:

```
rsync -av --exclude .git --exclude .venv --exclude 'pipeline/results' \
  ~/Claude/loraTeacher_git/ <cwl>@sockeye.arc.ubc.ca:/scratch/<alloc>/<cwl>/loraTeacher/
```

Copying, not cloning, also means nothing on the cluster can push to the team's repository. Run the same command again whenever you change code on your laptop.

## Step 4. Make the Python environment (login node, once)

**Alliance:**

```
module load StdEnv/2023 gcc python/3.11 cuda arrow opencv   # load BEFORE activating the venv; vLLM needs opencv >= 4.13 (module spider opencv)
virtualenv --no-download ~/venvs/subliminal
source ~/venvs/subliminal/bin/activate
pip install --no-index --upgrade pip
cd ~/scratch/loraTeacher/pipeline
pip install --no-index -r requirements.txt
```

`--no-index` installs from the Alliance's own wheel collection. If a package is missing there (`vllm` is the likely one), install it from PyPI instead: `pip install vllm` (the login node has internet). If vLLM cannot be installed at all, the pipeline runs with `--backend hf`, which is slower but needs nothing extra.

**Sockeye:**

```
module load miniconda3
conda create -y -p /arc/project/<alloc>/<cwl>/envs/subliminal python=3.11
conda activate /arc/project/<alloc>/<cwl>/envs/subliminal
cd /scratch/<alloc>/<cwl>/loraTeacher/pipeline
pip install -r requirements.txt
```

The environment lives on `/arc/project`, which compute nodes can read.

## Step 5. Download the models (login node, once)

```
export HF_HOME=~/scratch/hf                     # Sockeye: /scratch/<alloc>/<cwl>/hf
hf download Qwen/Qwen2.5-0.5B-Instruct
hf download Qwen/Qwen2.5-7B-Instruct   # about 15 GB
hf download meta-llama/Llama-3.1-8B-Instruct --exclude "original/*"   # gated: request access first; skips a duplicate 16 GB copy
hf download meta-llama/Llama-3.2-1B-Instruct --exclude "original/*"   # Llama smoke test
```

Alliance `~/scratch` is purged after a period of inactivity (check the cluster's "Scratch purging policy" page); keep anything you need long-term in `~/projects/def-<supervisor>/<user>/`.

## Step 6. Point the jobs at the environment (once)

```
cd <your copy>/pipeline
cp slurm/env.alliance.example.sh slurm/env.sh     # or env.sockeye.example.sh
nano slurm/env.sh                                 # fill in the paths from steps 4 and 5
```

## Step 7. Try things interactively

**Sockeye, in the browser:** log in to ARC OnDemand (CWL and MFA; VPN off campus) and start a Jupyter or VS Code session under Interactive Apps, with the GPU account, 1 GPU, 6 cores, 48 GB and 2 to 4 hours. ARC's Confluence page on OnDemand has the exact form fields. Open a terminal inside the session. Delete the session when you stop; it uses the allocation while it runs.

**Either cluster, from SSH:** ask for an interactive GPU shell:

```
salloc --account=def-<supervisor> --gres=gpu:1 --cpus-per-task=6 --mem=48G --time=2:00:00
# Sockeye: --account=st-<alloc>-1-gpu --gpus-per-node=1 --constraint=gpu_mem_32
```

Then, in that shell or the OnDemand terminal:

```
cd <your copy>/pipeline && source slurm/env.sh
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name(), 'bf16:', torch.cuda.is_bf16_supported())"
python scripts/run_stage.py --config configs/smoke.yaml --stage all
```

The smoke config runs the whole pipeline on the 0.5B model in minutes. If it fails inside vLLM, run it again with `--backend hf`, and use `--backend hf` (or `BACKEND=hf`) for every job on that cluster. To use the environment in Jupyter notebooks, run `pip install ipykernel && python -m ipykernel install --user --name subliminal` once and pick the `subliminal` kernel.

## Step 8. Copy in Erell's shared results (LR sweep only; skip for the cross-model study)

The sweep reuses whatever already exists with identical settings. From Erell's `results/main/`, copy into `<your copy>/pipeline/results/lr_sweep/`:

* `teachers/eagle__lora__tr16/` (the rank-16 eagle teacher, with its adapter),
* `datasets/eagle__lora__tr16__g1/`, `…__g2/`, `datasets/none__neutral__tr0__g1/`, `…__g2/` (the number data),
* if they exist, the students at learning rate 2e-4: `students/eagle__lora__tr16__sr{1,2,4}__g{1,2}__t1__ctxdb52f410/` and the same for `none__neutral__tr0`,
* `teachers.jsonl` and `datasets.jsonl`.

Use `rsync` or Globus (Alliance) from wherever Erell shares them. Everything that is missing is recomputed; nothing is overwritten, because the pipeline refuses to reuse a folder made with different settings.

## Step 9. Check the plan, then submit (shown for the LR sweep; the cross-model study uses its own configs below)

```
python scripts/estimate_budget.py --config configs/lr_sweep_eagle.yaml
python scripts/submit_cluster.py --cluster alliance --account def-<supervisor> \
  --config configs/lr_sweep_eagle.yaml --dry-run
```

The dry run prints the job arrays without submitting. If they look right, run the same command without `--dry-run`. It submits one job array per pipeline phase, chained so each starts when the previous one finishes, sized to tasks of about 6 hours, and sized again for only the unfinished work if you resubmit. On Sockeye use `--cluster sockeye --account st-<alloc>-1-gpu`. Ask for a specific GPU with `--gres`, for example `--gres gpu:a100:1` (names differ per cluster; `sinfo` shows them).

For a single stage without arrays, for example re-evaluating existing students, use the single-job script instead (edit its account and e-mail first):

```
sbatch --export=ALL,CONFIG=configs/context_mismatch.yaml,STAGE=4 slurm/sockeye_single.sbatch
```

## Step 10. Watch, fix, resume

```
squeue -u $USER                     # running and waiting jobs
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
seff <jobid>                        # how much of the requested memory and time a finished job used
tail -f results/lr_sweep/slurm_logs/<job>.out
```

If a task fails or runs out of time, fix the cause and run the same `submit_cluster.py` command again: finished students and evaluations are kept, and only the rest is submitted. Out of GPU memory on a V100 is the most likely failure for 7B training; the fix that keeps the effective batch at 66 is `batch_size: 11, grad_accum: 6`, but that is a settings change (it changes the fingerprint), so say so wherever the results are used.

## Step 11. Bring the results home and analyse them

On your laptop:

```
rsync -av --exclude 'students/*/adapter' --exclude 'teachers/*/adapter' \
  <user>@narval.alliancecan.ca:~/scratch/loraTeacher/pipeline/results/lr_sweep/ \
  ~/Claude/loraTeacher_git/pipeline/results/lr_sweep/
python analysis/lr_sweep.py --config pipeline/configs/lr_sweep_eagle.yaml
python analysis/specificity.py --config pipeline/configs/lr_sweep_eagle.yaml
```

The first command leaves the adapters on the cluster (they are large; copy them only for the alignment analysis). Put the adapters where the team agreed to keep them (16-bit is fine, per `docs/standards.md`), and add the results to `docs/results.md`.

## The cross-model and cross-category study (Qwen and Llama, new animals, trees)

Run it on an L40S cluster (48 GB, bf16; Erell's own runs also used L40S nodes among others), in this order. Every command runs from `pipeline/` after steps 1 to 7; steps 8 and 9 above are for the LR sweep and are not needed here, and steps 10 and 11 apply with `results/<config results_dir>/` in place of `results/lr_sweep/`. Each `submit_cluster.py` line is shown with `--dry-run` first. This study needs nothing from Erell: every model and target in it is new.

It does need two small additions to files in Erell's part of the pipeline. Finn's cluster copy has them, but they are not in the repository until Erell agrees:

* `src/subliminal/train_student.py` (in `load_base`, after the tokenizer is loaded) and `src/subliminal/generate.py` (in `HFSampler.__init__`, after the tokenizer is loaded): Llama ships without a padding token, so set it to the end-of-sequence token when it is missing. Qwen has its own padding token and is unaffected, so no existing run changes.
  ```python
  if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token
  ```
* `data/trait_prompts.yaml`: five more animals under `animals:` for the new-animal screens, `elephant: {plural: elephants}`, `tiger: {plural: tigers}`, `fox: {plural: foxes}`, `octopus: {plural: octopuses}` and `butterfly: {plural: butterflies}`. The tree configs use their own data folder, `data_trees/`, and need nothing here.

Without the first, the Llama configs fail; without the second, `qwen_animals_screen.yaml` and `llama_animals_screen.yaml` fail on the new animals. The Qwen tree screen needs neither.

1. Accept the Llama licence on the Hugging Face pages of `meta-llama/Llama-3.1-8B-Instruct` and `meta-llama/Llama-3.2-1B-Instruct`, run `hf auth login` on the login node, and download both (plus the two Qwen models) as in step 5.
2. Smoke tests, interactively (step 7): `configs/smoke.yaml` and `configs/smoke_llama.yaml`, `--stage all`. The Llama one checks the chat template and the padding fix; nothing on Llama has run anywhere before this.
3. The four screens, as batch jobs; they are independent, so submit all four:
   ```
   for c in qwen_animals_screen qwen_trees_screen llama_animals_screen llama_trees_screen; do
     python scripts/submit_cluster.py --cluster alliance --account aip-<supervisor> --gres gpu:l40s:1 \
       --max-parallel 20 --config configs/$c.yaml --dry-run
   done
   ```
   About 35 GPU-hours each by the pipeline's cost table (written for A100s; expect L40S to be similar or somewhat slower).
4. Look at each screen (`results/<screen>/figures/`, `tables/interaction.md`, `teachers.jsonl` for which LoRA teachers passed the gate). A target goes forward if its prompted or LoRA students clearly beat the neutral control at rank 8 or 64.
5. Put the 2 to 4 targets that transferred into `animals:` of the matching `*_confirm.yaml` and submit those the same way (about 43 GPU-hours for two targets, 77 for four).

## Resources and best practice

Ask for one GPU's fair share and no more. That is about 6 cores and 48 GB on Sockeye (4 V100s, 24 cores, about 192 GB per node). On the Alliance clusters, look up the cluster's "cores and memory per GPU" in its documentation (Narval: 12 cores and about 124 GB per A100, for example). Larger requests only wait longer. Ask for the time you expect plus about 50%. `submit_cluster.py` already sizes each task to about 6 hours, and every finished student is saved, so a task that times out loses only the item it was working on. Write logs to files named after the job (the scripts do), get an e-mail when a job ends or fails, and check a finished job with `seff` to tune the next request.
