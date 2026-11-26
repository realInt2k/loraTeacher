"""Nief-style figure with the learning rate tuned per student rank (as in Feng 2026).

For each (teacher, animal, student rank) we take the learning rate with the highest
mean transfer over seeds (Feng's choice; optimistic, since lr is picked on the same
runs). Dashed lines: the fixed default lr 2e-4 for comparison.

    python scripts/lr_tuned_figure.py --config configs/lr_sweep.yaml
"""

import argparse

import _path  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from subliminal.analysis import INK, INK_SECONDARY, SERIES, SURFACE, _style, base_rates, load_eval  # noqa: E402
from subliminal.config import load_config  # noqa: E402
from subliminal.runs import Store  # noqa: E402

DEFAULT_LR = 2e-4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    store = Store(cfg)
    bases = base_rates(load_eval(store), "p_substring")
    table = pd.read_csv(store.root / "tables" / "lr_sweep.csv")
    keys = ["animal", "teacher_kind", "teacher_rank", "student_rank"]
    best = table.loc[table.groupby(keys)["p_target"].idxmax()]
    fixed = table[(table["lr_value"] - DEFAULT_LR).abs() < 1e-9]
    best.to_csv(store.root / "tables" / "lr_tuned_best.csv", index=False)

    panels = [("Prompted teacher", "prompted")]
    panels += [(f"LoRA-finetuned teacher, teacher rank {r}", "lora") for r in sorted(table.loc[table.teacher_kind == "lora", "teacher_rank"].unique())]
    animals = sorted(table["animal"].unique())
    ranks = sorted(table["student_rank"].unique())
    fig, axes = plt.subplots(1, len(panels), figsize=(5.8 * len(panels), 4.6), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, (title, kind) in zip(axes[0], panels):
        _style(ax)
        for color, animal in zip(SERIES, animals):
            b = best[(best.animal == animal) & (best.teacher_kind == kind)].sort_values("student_rank")
            f = fixed[(fixed.animal == animal) & (fixed.teacher_kind == kind)].sort_values("student_rank")
            x, y, band = b.student_rank, b.p_target * 100, 1.96 * b["sem"] * 100
            ax.fill_between(x, y - band, y + band, color=color, alpha=0.15, linewidth=0)
            ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE)
            for xi, yi, lr in zip(x, y, b.lr_value):
                ax.annotate(f"{lr:g}", (xi, yi), textcoords="offset points", xytext=(0, 7), ha="center",
                            fontsize=6.5, color=color)  # fmt: skip
            ax.plot(f.student_rank, f.p_target * 100, color=color, linewidth=1.3, linestyle="--", alpha=0.6)
            if animal in bases:
                ax.axhline(bases[animal] * 100, color=color, linewidth=1.2, linestyle=(0, (2, 2)), alpha=0.9)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ranks)
        ax.set_xticklabels([str(r) for r in ranks])
        ax.minorticks_off()
        ax.set_ylim(0, 105)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("student LoRA rank", color=INK_SECONDARY, fontsize=9)
        ax.tick_params(labelleft=True)
    axes[0][0].set_ylabel("% responses containing target", color=INK_SECONDARY, fontsize=9)

    handles = [Line2D([], [], color=c, linewidth=2, marker="o") for c, _ in zip(SERIES, animals)]
    labels = [f"{a} (best lr per rank)" for a in animals]
    handles += [Line2D([], [], color=INK_SECONDARY, linewidth=1.3, linestyle="--"),
                Line2D([], [], color=INK_SECONDARY, linewidth=1.2, linestyle=(0, (2, 2)))]  # fmt: skip
    labels += [f"fixed lr {DEFAULT_LR:g}", "base model"]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9, labelcolor=INK)
    fig.text(0.01, 0.005, "Numbers above points: chosen lr. Shaded: ±1.96 SEM over seeds (n=2). "
             "lr chosen on the same runs (optimistic, as in Feng 2026).", fontsize=7.5, color=INK_SECONDARY)  # fmt: skip
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    path = store.root / "figures" / "nief_style_lr_tuned.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    print(path)


if __name__ == "__main__":
    main()
