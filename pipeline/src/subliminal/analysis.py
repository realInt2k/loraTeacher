"""Tables and figures (stage 6). Everything is read from the results store.

Outputs under ``results/<experiment>/``:

* ``figures/main_<variant>.png``: P(target) vs student rank, one line per teacher rank
* ``figures/alignment_heatmap_<metric>_<animal>.png``, ``figures/alignment_scatter.png``
* ``tables/interaction.csv|md``: teacher rank x student rank, mean P(target) with 95% CI
* ``tables/teacher_strength.csv|md``: each teacher's verified P(target)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Experiment
from .runs import NEUTRAL, PROMPTED, Store

# Presentation tokens (light surface). Categorical hues are assigned in fixed order.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL_GRAY = "#8a8985"
DIVERGING = ["#2a78d6", "#f0efec", "#e34948"]  # two hues around a neutral midpoint
SEQUENTIAL = ["#f0efec", "#86b6ef", "#2a78d6", "#0d366b"]

KEYS = ["animal", "teacher_kind", "teacher_rank", "student_rank"]


def bootstrap_ci(matrix: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI of the mean of a (seeds x questions) matrix, resampling both axes."""
    rng = np.random.default_rng(seed)
    n_seeds, n_questions = matrix.shape
    means = np.empty(n_boot)
    for i in range(n_boot):
        rows = rng.integers(0, n_seeds, n_seeds)
        cols = rng.integers(0, n_questions, n_questions)
        means[i] = matrix[np.ix_(rows, cols)].mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def load_eval(store: Store, context: str = "matched", prefix: bool = False) -> pd.DataFrame:
    """Eval rows for one context, de-duplicated (the last row for a key wins)."""
    frame = pd.DataFrame(store.read("eval"))
    if frame.empty:
        return frame
    frame = frame[(frame["eval_context"] == context) & (frame["number_prefix"] == prefix)]
    return frame.drop_duplicates(["model_id", "target"], keep="last").reset_index(drop=True)


def base_rates(frame: pd.DataFrame, metric: str) -> dict[str, float]:
    base = frame[frame["teacher_kind"] == "base"]
    return dict(zip(base["target"], base[metric]))


def student_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Student rows only, with ``animal`` set to the evaluated target (controls have none)."""
    students = frame[frame["student_rank"].notna()].copy()
    students["animal"] = students["target"]
    students["teacher_rank"] = students["teacher_rank"].fillna(0).astype(int)
    students["student_rank"] = students["student_rank"].astype(int)
    for column in ("generation_seed", "training_seed"):
        students[column] = students[column].astype(int)
    return students


def interaction_table(students: pd.DataFrame, metric: str = "p_substring") -> pd.DataFrame:
    """Mean P(target) per (animal, teacher kind, teacher rank, student rank) with CIs.

    ``sem`` is across seeds (for the mean +- 1.96 SEM bands); ``ci_low``/``ci_high``
    are bootstrap 95% CIs over seeds and over the evaluation questions.
    """
    per_question = "per_question_substring" if metric == "p_substring" else "per_question_word"
    records = []
    for key, group in students.groupby(KEYS, dropna=False):
        values = group[metric].to_numpy(dtype=float)
        matrix = np.array(group[per_question].tolist(), dtype=float)
        low, high = bootstrap_ci(matrix)
        sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan
        records.append(
            {**dict(zip(KEYS, key)), "mean": values.mean(), "sem": sem, "ci_low": low, "ci_high": high,
             "n_seeds": len(values), "metric": metric}
        )  # fmt: skip
    return pd.DataFrame(records)


def teacher_strength_table(store: Store, frame: pd.DataFrame) -> pd.DataFrame:
    """Verified P(target) for LoRA teachers plus the prompted teachers as reference."""
    records = []
    teachers = pd.DataFrame(store.read("teachers"))
    if not teachers.empty:
        for row in teachers.drop_duplicates("teacher_id", keep="last").to_dict("records"):
            for target, p_target in row["p_target"].items():
                records.append(
                    {
                        "animal": target,
                        "teacher_kind": row["teacher_kind"],
                        "teacher_rank": row["teacher_rank"],
                        "p_target": p_target,
                        "p_target_word": row["p_target_word"][target],
                        "p_target_number_prefix": row["p_target_number_prefix"][target],
                        "base_rate": row["base_rate"].get(target),
                        "threshold": row.get("threshold"),
                        "passed": row["passed"],
                        "number_shift_ratio": row.get("number_shift_ratio"),
                    }
                )
    all_rows = pd.DataFrame(store.read("eval"))
    if not all_rows.empty:
        prompted = all_rows[all_rows["eval_context"] == "trait_prompt"].drop_duplicates("model_id", keep="last")
        for row in prompted.to_dict("records"):
            records.append(
                {"animal": row["target"], "teacher_kind": PROMPTED, "teacher_rank": None,
                 "p_target": row["p_substring"], "p_target_word": row["p_word"], "passed": True}
            )  # fmt: skip
    return pd.DataFrame(records)


def _write_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path.with_suffix(".csv"), index=False)
    lines = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    for row in table.itertuples(index=False):
        cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def _line(ax, data: pd.DataFrame, color: str, label: str, linestyle: str = "-", marker: str = "o") -> None:
    data = data.sort_values("student_rank")
    x = data["student_rank"].to_numpy(dtype=float)
    mean = data["mean"].to_numpy(dtype=float) * 100
    ax.plot(x, mean, color=color, linewidth=2, linestyle=linestyle, marker=marker, markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)  # fmt: skip
    band = 1.96 * data["sem"].to_numpy(dtype=float) * 100
    if np.isfinite(band).any():
        band = np.nan_to_num(band)
        ax.fill_between(x, mean - band, mean + band, color=color, alpha=0.15, linewidth=0)


def main_figure(table: pd.DataFrame, bases: dict[str, float], variant: str, animals: list[str], path: Path) -> None:
    """P(target) vs student rank (log x): one line per teacher rank, one panel per animal.

    Overlays the prompted-teacher curve and the neutral-teacher control; the
    dashed horizontal line is the untouched base model. Bands are mean +- 1.96 SEM.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(animals), figsize=(4.2 * len(animals), 3.8), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, animal in zip(axes[0], animals):
        _style(ax)
        panel = table[(table["animal"] == animal) & (table["student_rank"] > 0)]
        lora = panel[panel["teacher_kind"] == variant]
        for color, rank in zip(SERIES, sorted(lora["teacher_rank"].unique())):
            _line(ax, lora[lora["teacher_rank"] == rank], color, f"LoRA teacher r={rank}")
        prompted = panel[panel["teacher_kind"] == PROMPTED]
        if not prompted.empty:
            _line(ax, prompted, INK, "prompted teacher", linestyle="--", marker="s")
        neutral = panel[panel["teacher_kind"] == NEUTRAL]
        if not neutral.empty:
            _line(ax, neutral, NEUTRAL_GRAY, "neutral teacher (control)", linestyle=":", marker="^")
        if animal in bases:
            ax.axhline(bases[animal] * 100, color=INK_SECONDARY, linewidth=1, linestyle=(0, (4, 3)),
                       label="base model")  # fmt: skip
        ranks = sorted(panel["student_rank"].unique())
        ax.set_xscale("log", base=2)
        if ranks:
            ax.set_xticks(ranks)
            ax.set_xticklabels([str(r) for r in ranks])
        ax.minorticks_off()
        ax.set_title(animal, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("student LoRA rank", color=INK_SECONDARY, fontsize=9)
    axes[0][0].set_ylabel("% responses containing target", color=INK_SECONDARY, fontsize=9)
    axes[0][0].set_ylim(bottom=0)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.5, -0.02))  # fmt: skip
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def _per_rank(runs: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean and SEM over all runs (seeds, and teacher ranks if several) at each student rank."""
    grouped = runs.groupby("student_rank")[metric]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"])
    return out


def nief_style_figure(
    students: pd.DataFrame,
    bases: dict[str, float],
    animals: list[str],
    lora_kind: str,
    path: Path,
    metric: str = "p_substring",
    split_teacher_ranks: bool = True,
) -> None:
    """Our results in the format of Nief et al. Figure 2.

    One line per animal, x = student LoRA rank (log scale), shaded mean +- 1.96 SEM
    across runs, dotted line = base model in the animal's colour. First panel:
    prompted teachers (the replication). Then either one panel per LoRA teacher
    rank (``split_teacher_ranks``, SEM over seeds) or one panel pooling all
    teacher ranks (SEM over teacher ranks x seeds).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    lora = students[students["teacher_kind"] == lora_kind]
    lora_ranks = sorted(int(r) for r in lora["teacher_rank"].unique())
    panels: list[tuple[str, pd.DataFrame]] = [
        ("Prompted teacher (as in Nief et al.)", students[students["teacher_kind"] == PROMPTED])
    ]
    if split_teacher_ranks:
        panels += [(f"LoRA-finetuned teacher, teacher rank {r}", lora[lora["teacher_rank"] == r]) for r in lora_ranks]
    else:
        panels.append((f"LoRA-finetuned teacher (mean over teacher ranks {', '.join(map(str, lora_ranks))})", lora))

    n_cols = 2
    n_rows = -(-len(panels) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11.5, 4.1 * n_rows + 0.4), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    all_ranks = sorted(int(r) for r in students["student_rank"].unique() if r > 0)
    for index, ax in enumerate(axes.flat):
        if index >= len(panels):
            ax.set_visible(False)
            continue
        title, subset = panels[index]
        _style(ax)
        for color, animal in zip(SERIES, animals):
            runs = subset[(subset["animal"] == animal) & (subset["student_rank"] > 0)]
            if runs.empty:
                continue
            stats = _per_rank(runs, metric).sort_values("student_rank")
            x = stats["student_rank"].to_numpy(dtype=float)
            mean = stats["mean"].to_numpy(dtype=float) * 100
            band = np.nan_to_num(1.96 * stats["sem"].to_numpy(dtype=float) * 100)
            ax.fill_between(x, mean - band, mean + band, color=color, alpha=0.15, linewidth=0)
            ax.plot(x, mean, color=color, linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=animal)  # fmt: skip
            if animal in bases:
                ax.axhline(bases[animal] * 100, color=color, linewidth=1.2, linestyle=(0, (2, 2)), alpha=0.9)
        ax.set_xscale("log", base=2)
        ax.set_xticks(all_ranks)
        ax.set_xticklabels([str(r) for r in all_ranks])
        ax.minorticks_off()
        ax.set_ylim(0, 100)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("student LoRA rank", color=INK_SECONDARY, fontsize=9)
        if index % n_cols == 0:
            ax.set_ylabel("% responses containing target", color=INK_SECONDARY, fontsize=9)
        ax.tick_params(labelleft=True)

    handles = [Line2D([], [], color=color, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE)
               for color, _ in zip(SERIES, animals)]  # fmt: skip
    labels = list(animals)
    handles.append(Line2D([], [], color=INK_SECONDARY, linewidth=1.2, linestyle=(0, (2, 2))))
    labels.append("base model (dotted, animal colour)")
    legend_height = 0.55 / (4.1 * n_rows + 0.4)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.5, 0.0))  # fmt: skip
    fig.text(0.01, 0.003, "Shaded: mean ± 1.96 SEM across runs.", fontsize=8, color=INK_SECONDARY)
    fig.tight_layout(rect=(0, legend_height, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def alignment_heatmaps(alignment: pd.DataFrame, animal: str, metric: str, path: Path) -> None:
    """Layer x module heatmaps of teacher-vs-student alignment, one panel per (teacher, student) rank."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    from .alignment import MODULES

    data = alignment[
        (alignment["comparison"] == "teacher_vs_student") & (alignment["animal"] == animal) & (alignment["layer"] != "all")
    ].copy()
    if data.empty:
        return
    data["layer"] = data["layer"].astype(int)
    teacher_ranks = sorted(data["teacher_rank"].unique())
    student_ranks = sorted(data["student_rank"].unique())
    diverging = metric == "cosine"
    cmap = LinearSegmentedColormap.from_list("m", DIVERGING if diverging else SEQUENTIAL)
    limit = float(np.nanmax(np.abs(data[metric]))) or 1.0
    vmin, vmax = (-limit, limit) if diverging else (0.0, limit)

    fig, axes = plt.subplots(len(teacher_ranks), len(student_ranks), squeeze=False, sharex=True, sharey=True,
                             figsize=(1.9 * len(student_ranks) + 1.2, 2.6 * len(teacher_ranks) + 0.8))  # fmt: skip
    fig.patch.set_facecolor(SURFACE)
    image = None
    for i, teacher_rank in enumerate(teacher_ranks):
        for j, student_rank in enumerate(student_ranks):
            ax = axes[i][j]
            cell = data[(data["teacher_rank"] == teacher_rank) & (data["student_rank"] == student_rank)]
            grid = cell.pivot_table(index="layer", columns="module", values=metric, aggfunc="mean")
            grid = grid.reindex(columns=[m for m in MODULES])
            image = ax.imshow(grid.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                              origin="lower", interpolation="nearest")  # fmt: skip
            ax.set_xticks(range(len(MODULES)))
            ax.set_xticklabels([m.replace("_proj", "") for m in MODULES], rotation=90, fontsize=7, color=INK_SECONDARY)
            ax.tick_params(colors=INK_SECONDARY, labelsize=7)
            if i == 0:
                ax.set_title(f"student r={student_rank}", fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"teacher r={teacher_rank}\nlayer", fontsize=9, color=INK)
    if image is not None:
        bar = fig.colorbar(image, ax=axes, shrink=0.8, pad=0.02)
        bar.set_label(metric, color=INK_SECONDARY, fontsize=9)
        bar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    fig.suptitle(f"{animal}: teacher vs student update, {metric} (mean over seeds)", fontsize=11, color=INK, x=0.02, ha="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def alignment_scatter(alignment: pd.DataFrame, students: pd.DataFrame, metric: str, path: Path) -> None:
    """Global teacher-student cosine vs P(target), one point per student; neutral students are the null."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    globals_ = alignment[alignment["layer"] == "all"]
    seeds = ["generation_seed", "training_seed"]
    trait = globals_[globals_["comparison"] == "teacher_vs_student"].merge(
        students[KEYS + seeds + [metric]], on=KEYS + seeds, how="inner"
    )
    # Null: the labelled run's teacher against a neutral-teacher student of the same rank and seeds.
    neutral = students[students["teacher_kind"] == NEUTRAL][["animal", "student_rank", *seeds, metric]]
    null = globals_[globals_["comparison"] == "teacher_vs_neutral_student"].merge(
        neutral, on=["animal", "student_rank", *seeds], how="inner"
    )
    if trait.empty and null.empty:
        return
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    if not null.empty:
        ax.scatter(null["cosine"], null[metric] * 100, s=36, color=NEUTRAL_GRAY, marker="^", edgecolor=SURFACE,
                   linewidth=1, label="neutral-teacher student (null)")  # fmt: skip
    for color, rank in zip(SERIES, sorted(trait["teacher_rank"].unique())):
        part = trait[trait["teacher_rank"] == rank]
        ax.scatter(part["cosine"], part[metric] * 100, s=40, color=color, edgecolor=SURFACE, linewidth=1,
                   label=f"teacher r={rank}")  # fmt: skip
    ax.set_xlabel("global cosine(teacher update, student update)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("% responses containing target", color=INK_SECONDARY, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def make_all(cfg: Experiment, store: Store, metric: str = "p_substring") -> list[Path]:
    """Regenerate every table and figure from the results store."""
    written: list[Path] = []
    frame = load_eval(store)
    if frame.empty:
        raise RuntimeError(f"no evaluation results in {store.root}; run stage 4 first")
    students = student_frame(frame)
    bases = base_rates(frame, metric)
    tables = store.root / "tables"
    figures = store.root / "figures"

    table = interaction_table(students, metric)
    _write_table(table, tables / "interaction")
    strength = teacher_strength_table(store, frame)
    if not strength.empty:
        _write_table(strength, tables / "teacher_strength")
    written += [tables / "interaction.csv", tables / "teacher_strength.csv"]

    variants = [v.name for v in cfg.teacher.variants if v.has_trait] or ["lora"]
    for variant in variants:
        animals = [a for a in cfg.animals if not table[(table["animal"] == a)].empty]
        path = figures / f"main_{variant}.png"
        main_figure(table, bases, variant, animals, path)
        written.append(path)
    if (students["teacher_kind"] == variants[0]).any() or (students["teacher_kind"] == PROMPTED).any():
        animals = [a for a in cfg.animals if (students["animal"] == a).any()]
        path = figures / "nief_style.png"  # one panel per teacher rank
        nief_style_figure(students, bases, animals, variants[0], path, metric, split_teacher_ranks=True)
        written.append(path)
        path = figures / "nief_style_pooled.png"  # teacher ranks averaged
        nief_style_figure(students, bases, animals, variants[0], path, metric, split_teacher_ranks=False)
        written.append(path)

    alignment_path = store.root / "alignment.csv"
    if alignment_path.exists():
        alignment = pd.read_csv(alignment_path, dtype={"layer": str})
        if not alignment.empty:
            for animal in cfg.animals:
                for quantity in ("cosine", "proj_both"):
                    path = figures / f"alignment_heatmap_{quantity}_{animal}.png"
                    alignment_heatmaps(alignment, animal, quantity, path)
                    written.append(path)
            path = figures / "alignment_scatter.png"
            alignment_scatter(alignment, students, metric, path)
            written.append(path)
    return written
