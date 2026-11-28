import os, shutil, torch
from safetensors.torch import save_file
from common import RANK, OUT_DIR, SAVE_DIR, STEP_FILE

with open(STEP_FILE, "r", encoding="utf-8") as file:
    checkpoint = file.read().strip()
    print(checkpoint)

#checkpoint = 745
if not checkpoint:
    print('bad')
    exit(-1)
src = f"{OUT_DIR}/checkpoint-{checkpoint}"
sd = torch.load(f"{src}/pytorch_model_fsdp.bin", map_location="cpu", weights_only=True)
print(len(sd), list(sd)[:2])

out = {}
for k, v in sd.items():
    if "lora_" not in k:
        continue
    k = k.replace("_checkpoint_wrapped_module.", "").replace(".default", "")
    out[k] = v.detach().contiguous()
print(len(out), next(iter(out)), next(iter(out.values())).shape)   # expect 392, ..., lora_A [128, 3584]

os.makedirs(f"{SAVE_DIR}", exist_ok=True)
save_file(out, f"{SAVE_DIR}/adapter_model.safetensors", metadata={"format": "pt"})
shutil.copy(f"{src}/adapter_config.json", f"{SAVE_DIR}/adapter_config.json")
