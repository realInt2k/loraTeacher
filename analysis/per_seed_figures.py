"""Main figures with every data seed shown as a point (experiment 8 in docs/experiments.md).

    python analysis/per_seed_figures.py --config pipeline/configs/main_teacher_x_student.yaml
    python analysis/per_seed_figures.py --config ... --metric p_word --context empty

The pipeline's figures (stage 6) draw the mean over seeds with a band of
+-1.96 SEM. With three seeds that band is wide and says little about what the
seeds did: one outlier seed and three moderately different seeds look the same.
These figures draw the same means as lines, and each seed's own rate as a dot
next to it, so wide bands can be read. They use the pipeline's own loaders
(subliminal.analysis.load_eval, student_frame, base_rates) and style, read the
results store named in the config, and write next to the pipeline's figures:

* figures/per_seed_main_<variant>.png: one panel per animal, one line per
  teacher rank, plus the prompted teacher and the neutral-teacher control
  (the layout of figures/main_<variant>.png).
* figures/per_seed_nief_style.png: one panel per teacher type and rank, one
  line per animal (the layout of figures/nief_style.png).
* tables/per_seed.csv: the plotted values, one row per student and seed.

Only reads the store; never changes it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline" / "src"))

from subliminal import analysis as A  # noqa: E402
from subliminal.config import load_config  # noqa: E402
from subliminal.runs import NEUTRAL, PROMPTED, Store  # noqa: E402

SEED_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]  # one marker per data seed


def _offsets(n: int, spread: float = 0.10) -> list[float]:
    """Horizontal offsets (in log2 units) so the points of different seeds do not overlap."""
    if n <= 1:
        return [0.0]
    return list(np.linspace(-spread, spread, n))


def draw_series(ax, runs: pd.DataFrame, metric: str, color: str, label: str, seeds: list[int],
                linestyle: str = "-", series_shift: float = 0.0) -> None:  # fmt: skip
    """Mean over seeds as a line; each seed's value as a small dot beside it."""
    if runs.empty:
        return
    means = runs.groupby("student_rank")[metric].mean().sort_index()
    x = means.index.to_numpy(dtype=float)
    ax.plot(x * 2**series_shift, means.to_numpy() * 100, color=color, linewidth=1.8, linestyle=linestyle,
            label=label, zorder=3)  # fmt: skip
    for offset, seed, marker in zip(_offsets(len(seeds)), seeds, SEED_MARKERS):
        part = runs[runs["generation_seed"] == seed]
        if part.empty:
            continue
        # Several teacher ranks pooled in one series give several points per seed; draw them all.
        ax.scatter(part["student_rank"].to_numpy(dtype=float) * 2 ** (series_shift + offset),
                   part[metric].to_numpy(dtype=float) * 100, s=16, marker=marker, color=color,
                   alpha=0.75, edgecolors=A.SURFACE, linewidths=0.5, zorder=4)  # fmt: skip


def _axes_ranks(ax, ranks: list[int]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.minorticks_off()
    ax.set_xlabel("student LoRA rank", color=A.INK_SECONDARY, fontsize=9)


def _seed_legend(fig, seeds: list[int]) -> None:
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=A.INK_SECONDARY, marker=m, linestyle="", markersize=5)
               for _, m in zip(seeds, SEED_MARKERS)]  # fmt: skip
    fig.legend(handles, [f"data seed {s}" for s in seeds], loc="upper right", frameon=False, fontsize=8,
               labelcolor=A.INK_SECONDARY, ncol=len(seeds))  # fmt: skip


def main_layout(students, bases, variant, animals, metric, seeds, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(animals), figsize=(4.2 * len(animals), 4.0), sharey=True, squeeze=False)
    fig.patch.set_facecolor(A.SURFACE)
    ranks = sorted(int(r) for r in students["student_rank"].unique() if r > 0)
    for ax, animal in zip(axes[0], animals):
        A._style(ax)
        panel = students[(students["animal"] == animal) & (students["student_rank"] > 0)]
        lora = panel[panel["teacher_kind"] == variant]
        teacher_ranks = sorted(lora["teacher_rank"].unique())
        shifts = _offsets(len(teacher_ranks) + 2, spread=0.22)  # keep the series' dots apart too
        for i, (color, rank) in enumerate(zip(A.SERIES, teacher_ranks)):
            draw_series(ax, lora[lora["teacher_rank"] == rank], metric, color, f"LoRA teacher r={rank}", seeds,
                        series_shift=shifts[i])  # fmt: skip
        draw_series(ax, panel[panel["teacher_kind"] == PROMPTED], metric, A.INK, "prompted teacher", seeds,
                    linestyle="--", series_shift=shifts[-2])  # fmt: skip
        draw_series(ax, panel[panel["teacher_kind"] == NEUTRAL], metric, A.NEUTRAL_GRAY,
                    "neutral teacher (control)", seeds, linestyle=":", series_shift=shifts[-1])  # fmt: skip
        if animal in bases:
            ax.axhline(bases[animal] * 100, color=A.INK_SECONDARY, linewidth=1, linestyle=(0, (4, 3)),
                       label="base model")  # fmt: skip
        _axes_ranks(ax, ranks)
        ax.set_title(animal, color=A.INK, fontsize=11, loc="left")
    axes[0][0].set_ylabel(_ylabel(metric), color=A.INK_SECONDARY, fontsize=9)
    axes[0][0].set_ylim(bottom=0)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), frameon=False, fontsize=9,
               labelcolor=A.INK, bbox_to_anchor=(0.5, -0.02))  # fmt: skip
    _seed_legend(fig, seeds)
    fig.text(0.01, 0.975, "Line: mean over seeds. Dots: individual data seeds (shifted sideways to stay apart).",
             fontsize=8, color=A.INK_SECONDARY, va="top")  # fmt: skip
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    _save(fig, path)


def nief_layout(students, bases, variant, animals, metric, seeds, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lora = students[students["teacher_kind"] == variant]
    panels = [("Prompted teacher", students[students["teacher_kind"] == PROMPTED])]
    panels += [(f"LoRA-finetuned teacher, teacher rank {int(r)}", lora[lora["teacher_rank"] == r])
               for r in sorted(lora["teacher_rank"].unique())]  # fmt: skip
    n_cols = 2
    n_rows = -(-len(panels) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11.5, 4.1 * n_rows + 0.6), sharey=True, squeeze=False)
    fig.patch.set_facecolor(A.SURFACE)
    ranks = sorted(int(r) for r in students["student_rank"].unique() if r > 0)
    shifts = _offsets(len(animals), spread=0.22)
    for index, ax in enumerate(axes.flat):
        if index >= len(panels):
            ax.set_visible(False)
            continue
        title, subset = panels[index]
        A._style(ax)
        for i, (color, animal) in enumerate(zip(A.SERIES, animals)):
            runs = subset[(subset["animal"] == animal) & (subset["student_rank"] > 0)]
            draw_series(ax, runs, metric, color, animal, seeds, series_shift=shifts[i])
            if animal in bases and not runs.empty:
                ax.axhline(bases[animal] * 100, color=color, linewidth=1.1, linestyle=(0, (2, 2)), alpha=0.9)
        _axes_ranks(ax, ranks)
        ax.set_ylim(0, 100)
        ax.set_title(title, color=A.INK, fontsize=11, loc="left")
        if index % n_cols == 0:
            ax.set_ylabel(_ylabel(metric), color=A.INK_SECONDARY, fontsize=9)
        ax.tick_params(labelleft=True)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9,
               labelcolor=A.INK, bbox_to_anchor=(0.5, 0.0))  # fmt: skip
    _seed_legend(fig, seeds)
    fig.text(0.01, 0.003, "Line: mean over seeds. Dots: individual data seeds. Dotted: base model.",
             fontsize=8, color=A.INK_SECONDARY)  # fmt: skip
    fig.tight_layout(rect=(0, 0.6 / (4.1 * n_rows + 0.6), 1, 0.96))
    _save(fig, path)


def _ylabel(metric: str) -> str:
    return "% responses containing target" if metric == "p_substring" else "% responses naming target (word)"


def _save(fig, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=A.SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="pipeline config whose results store to read")
    ap.add_argument("--metric", default="p_substring", choices=["p_substring", "p_word"])
    ap.add_argument("--context", default="matched", help="evaluation context (matched, empty, chatgpt, ...)")
    ap.add_argument("--results-dir", help="read this results dir instead of the config's (e.g. a local copy)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.results_dir:
        cfg = cfg.model_copy(update={"results_dir": args.results_dir})
    store = Store(cfg)
    frame = A.load_eval(store, context=args.context)
    if frame.empty:
        sys.exit(f"no evaluation rows for context '{args.context}' in {store.root}")
    students = A.student_frame(frame)
    bases = A.base_rates(frame, args.metric)
    seeds = sorted(int(s) for s in students["generation_seed"].unique())
    suffix = "" if (args.metric, args.context) == ("p_substring", "matched") else f"_{args.metric}_{args.context}"

    tables = store.root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    columns = ["animal", "teacher_kind", "teacher_rank", "student_rank", "generation_seed", "training_seed",
               "run_id", args.metric]  # fmt: skip
    students[[c for c in columns if c in students]].sort_values(columns[:5]).to_csv(
        tables / f"per_seed{suffix}.csv", index=False
    )

    written = [tables / f"per_seed{suffix}.csv"]
    variants = [v.name for v in cfg.teacher.variants if v.has_trait] or ["lora"]
    animals = [a for a in cfg.animals if (students["animal"] == a).any()]
    for variant in variants:
        if not (students["teacher_kind"] == variant).any():
            continue
        path = store.root / "figures" / f"per_seed_main_{variant}{suffix}.png"
        main_layout(students, bases, variant, animals, args.metric, seeds, path)
        written.append(path)
    path = store.root / "figures" / f"per_seed_nief_style{suffix}.png"
    nief_layout(students, bases, variants[0], animals, args.metric, seeds, path)
    written.append(path)
    counts = students.groupby(["teacher_kind", "student_rank"])["generation_seed"].nunique()
    if (counts < 3).any():
        print("note: some cells have fewer than 3 seeds:\n" + counts[counts < 3].to_string())
    print("\n".join(str(p) for p in written))


if __name__ == "__main__":
    main()
