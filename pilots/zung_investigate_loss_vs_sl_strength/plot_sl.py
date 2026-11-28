"""Usage: python plot_sl.py penguin_rank4.json [out.png]"""
import json, sys
import matplotlib.pyplot as plt

d = json.load(open(sys.argv[1]))
out = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].replace(".json", ".png")
animal = d["teacher_animal"]
ck = d["checkpoints"]

ep = [c["epoch"] for c in ck]
st = [100 * c["animals"][animal]["student"] for c in ck]
base = 100 * ck[0]["animals"][animal]["base"]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(ep, st, color="black", marker="o", ms=4, label=f"student ({animal})")
ax.axhline(base, color="black", ls="--", label=f"student (base {d['base_model']})")
ax.set_xlabel("epoch")
ax.set_ylabel("SL strength (%)")
ax.set_ylim(0, max(st) * 1.4)
ax.set_title(f"{animal.capitalize()} teacher, LoRA rank {d['lora_rank']} student, lr {d['learning_rate']:.0e}".replace("e-0", "e-"))

ax2 = ax.twinx()
ax2.plot(ep, [c["train_loss"] for c in ck], color="blue", ls="--", label="train loss")
ax2.plot(ep, [c["val_loss"] for c in ck], color="red", ls="--", label="val loss")
ax2.set_ylim(0, 2)
ax2.set_ylabel("loss")

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(out, dpi=150)
print("saved", out)
