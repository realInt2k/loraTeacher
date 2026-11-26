"""Is transfer specific to the teacher's animal? (experiment 6 in docs/experiments.md)

    python analysis/specificity.py --config pipeline/configs/main_teacher_x_student.yaml
    python analysis/specificity.py --config ... --rule family --context matched

For every student, count how often it names *each* animal (analysis/animals.py),
subtract the same rate in its matched control student, and average over seeds
(and teacher ranks). Arranged as a matrix, rows = the teacher's animal,
columns = what the student says:

* a dominant diagonal means students pick up their teacher's animal specifically;
* a column that rises in every row (panda, phoenix, ...) means a general
  reshuffling of answers that does not depend on the teacher.

The matched control has the same student rank, data seed, training seed and
context. For students of LoRA teachers it is the student of the fine-tuned
neutral teacher (lora_neutral) when one exists at that student rank, otherwise
the student of the neutral teacher; for prompted teachers it is always the
neutral teacher's student. The output says which control each cell used.

Reads the raw answers in <results>/eval/*.json.gz (the pipeline's own layout,
nothing is converted) and writes to <results>/specificity/:

* matrix_<kind>_sr<rank>.csv / .png    one matrix per teacher type and student rank
* matrix_<kind>_low_vs_high.png         low ranks (<=8) and high ranks (>=64) side by side
* per_student.csv                       every student's rates, control rates and deltas
* summary.md                            diagonal vs off-diagonal, the panda column, entropy

Intervals are t-intervals across data seeds (the pipeline's figures use the
same seeds as the unit of replication).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline" / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

import animals as L  # noqa: E402
from subliminal import analysis as A  # noqa: E402
from subliminal.config import load_config  # noqa: E402
from subliminal.runs import NEUTRAL, PROMPTED, Store  # noqa: E402

CONTROL_KINDS = (NEUTRAL, "lora_neutral")
KEY = ["student_rank", "generation_seed", "training_seed", "context_id"]


def load_students(store: Store, context: str, rule: str) -> tuple[pd.DataFrame, dict | None]:
    """One row per student evaluated in ``context``: meta columns, rate_<animal>, entropy."""
    rows, base = [], None
    for path in sorted((store.root / "eval").glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            raw = json.load(handle)
        meta = raw["meta"]
        if meta.get("eval_context") != context or meta.get("number_prefix"):
            continue
        answers = [a for per_question in raw["responses"] for a in per_question]
        counts = L.distribution(answers, rule)
        n = len(answers)
        rates = {f"rate_{a}": counts.get(a, 0) / n for a in L.ANIMALS}
        rates["rate_other"] = counts.get("other", 0) / n
        record = {**meta, **rates, "entropy": L.entropy_bits(counts), "n_answers": n}
        if meta.get("student_rank") is None:
            if meta.get("model_id") == "base":
                base = record
            continue
        rows.append(record)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["teacher_rank"] = frame["teacher_rank"].fillna(0).astype(int)
        for column in ("student_rank", "generation_seed", "training_seed"):
            frame[column] = frame[column].astype(int)
    return frame, base


def attach_controls(frame: pd.DataFrame) -> pd.DataFrame:
    """Add control_kind and delta_<animal> (student minus matched control) to every trait student."""
    controls = {kind: frame[frame["teacher_kind"] == kind].set_index(KEY) for kind in CONTROL_KINDS}
    rate_columns = [c for c in frame.columns if c.startswith("rate_")] + ["entropy"]
    out = []
    for _, row in frame[~frame["teacher_kind"].isin(CONTROL_KINDS)].iterrows():
        key = tuple(row[k] for k in KEY)
        order = ["lora_neutral", NEUTRAL] if str(row["teacher_kind"]).startswith("lora") else [NEUTRAL]
        match, used = None, None
        for kind in order:
            table = controls[kind]
            if key in table.index:
                match = table.loc[[key]].iloc[-1]
                used = kind
                break
        record = row.to_dict()
        record["control_kind"] = used
        for column in rate_columns:
            record[f"delta_{column.removeprefix('rate_')}"] = (row[column] - match[column]) if used else np.nan
        out.append(record)
    return pd.DataFrame(out)


def t_interval(values: np.ndarray) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    mean = float(values.mean())
    if len(values) < 2:
        return mean, np.nan, np.nan
    from scipy import stats

    half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return mean, mean - half, mean + half


def matrix(students: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Mean delta, rows = teacher animal, columns = answered animal (pooled over seeds and teacher ranks)."""
    rows = {}
    for animal, group in students.groupby("animal"):
        rows[animal] = {c: group[f"delta_{c}"].mean() for c in columns}
    return pd.DataFrame(rows).T


def answer_columns(students: pd.DataFrame, teacher_animals: list[str], extra: int = 6) -> list[str]:
    """Teacher animals first (so the diagonal is the diagonal), then the animals that moved most."""
    movement = {a: students[f"delta_{a}"].abs().mean() for a in L.ANIMALS if a not in teacher_animals}
    top = [a for a, _ in sorted(movement.items(), key=lambda kv: -kv[1])[:extra]]
    if "panda" not in top and "panda" not in teacher_animals:
        top = top[:-1] + ["panda"]
    return teacher_animals + top


def heatmap(ax, mat: pd.DataFrame, title: str, vmax: float) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("div", A.DIVERGING)
    values = mat.to_numpy(dtype=float) * 100
    image = ax.imshow(values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(mat.shape[1]), [c.replace("_", " ") for c in mat.columns], rotation=45, ha="right",
                  fontsize=8)  # fmt: skip
    ax.set_yticks(range(mat.shape[0]), mat.index, fontsize=9)
    ax.set_xlabel("what the student says", fontsize=9, color=A.INK_SECONDARY)
    ax.set_ylabel("teacher's animal", fontsize=9, color=A.INK_SECONDARY)
    ax.set_title(title, fontsize=10, loc="left", color=A.INK)
    for (i, j), v in np.ndenumerate(values):
        if np.isfinite(v):
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=7,
                    color=A.SURFACE if abs(v) > 0.6 * vmax else A.INK)  # fmt: skip
    return image


def figure(panels: list[tuple[str, pd.DataFrame]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmax = max(5.0, max(np.nanmax(np.abs(m.to_numpy(dtype=float))) * 100 for _, m in panels))
    fig, axes = plt.subplots(1, len(panels), figsize=(0.55 * panels[0][1].shape[1] * len(panels) + 1.5, 3.6),
                             squeeze=False)  # fmt: skip
    fig.patch.set_facecolor(A.SURFACE)
    image = None
    for ax, (title, mat) in zip(axes[0], panels):
        image = heatmap(ax, mat, title, vmax)
    bar = fig.colorbar(image, ax=axes[0].tolist(), shrink=0.8)
    bar.set_label("change vs matched control (percentage points)", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=A.SURFACE, bbox_inches="tight")
    plt.close(fig)


def summary(trait: pd.DataFrame, controls: pd.DataFrame, base: dict | None, rule: str, context: str) -> str:
    lines = [f"# Specificity of transfer ({rule} counting rule, eval context {context})", "",
             "Each value is a mean over data seeds (and teacher ranks where several are pooled), "
             "with a 95% t-interval across seeds in brackets; percentage points against the matched "
             "control student.", ""]  # fmt: skip

    def cell(values):
        m, lo, hi = t_interval(np.asarray(values, dtype=float) * 100)
        return f"{m:+.1f}" + ("" if np.isnan(lo) else f" [{lo:+.1f}, {hi:+.1f}]")

    lines += ["## Own animal vs other teachers' animals", "",
              "Diagonal: rise of the teacher's own animal. Off-diagonal: mean rise of the *other* "
              "teacher animals in the same students. Transfer is specific where the first clearly "
              "exceeds the second.", "",
              "| teacher | student rank | n | own animal | other teacher animals | panda | entropy (bits) | control |",
              "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    teacher_animals = sorted(trait["animal"].unique())
    for (kind, rank), group in trait.groupby(["teacher_kind", "student_rank"]):
        own = [r[f"delta_{r['animal']}"] for _, r in group.iterrows()]
        others = [np.mean([r[f"delta_{a}"] for a in teacher_animals if a != r["animal"]]) for _, r in group.iterrows()]
        m, lo, hi = t_interval(group["delta_entropy"].to_numpy(dtype=float))
        entropy = f"{m:+.2f}" + ("" if np.isnan(lo) else f" [{lo:+.2f}, {hi:+.2f}]")
        used = ",".join(sorted(set(group["control_kind"].dropna())))
        lines.append(f"| {kind} | {rank} | {len(group)} | {cell(own)} | {cell(others)} | "
                     f"{cell(group['delta_panda'])} | {entropy} | {used} |")  # fmt: skip
    if not controls.empty and base is not None:
        lines += ["", "## The controls themselves against the base model", "",
                  "If the control students already move toward panda or lose entropy, that part comes "
                  "from training on numbers at that rank, not from any teacher's preference.", "",
                  "| control | student rank | n | panda, change vs base | entropy, change vs base |",
                  "|---|---|---|---|---|"]  # fmt: skip
        for (kind, rank), group in controls.groupby(["teacher_kind", "student_rank"]):
            panda = cell(group["rate_panda"] - base["rate_panda"])
            m, lo, hi = t_interval(group["entropy"].to_numpy(dtype=float) - base["entropy"])
            entropy = f"{m:+.2f}" + ("" if np.isnan(lo) else f" [{lo:+.2f}, {hi:+.2f}]")
            lines.append(f"| {kind} | {rank} | {len(group)} | {panda} | {entropy} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="pipeline config whose results store to read")
    ap.add_argument("--results-dir", help="read this results dir instead of the config's (e.g. a local copy)")
    ap.add_argument("--rule", default="standard", choices=["standard", "family"])
    ap.add_argument("--context", default="matched")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.results_dir:
        cfg = cfg.model_copy(update={"results_dir": args.results_dir})
    store = Store(cfg)
    frame, base = load_students(store, args.context, args.rule)
    if frame.empty:
        sys.exit(f"no student evaluations for context '{args.context}' in {store.root / 'eval'}")
    trait = attach_controls(frame)
    controls = frame[frame["teacher_kind"].isin(CONTROL_KINDS)]
    out = store.root / "specificity" / f"{args.rule}_{args.context}"
    out.mkdir(parents=True, exist_ok=True)
    trait.to_csv(out / "per_student.csv", index=False)
    (out / "summary.md").write_text(summary(trait, controls, base, args.rule, args.context))

    usable = trait[trait["control_kind"].notna()]
    teacher_animals = [a for a in cfg.animals if (usable["animal"] == a).any()]
    columns = answer_columns(usable, teacher_animals)
    written = [out / "summary.md", out / "per_student.csv"]
    for kind, by_kind in usable.groupby("teacher_kind"):
        label = "prompted teacher" if kind == PROMPTED else f"{kind} teacher (all teacher ranks)"
        for rank, group in by_kind.groupby("student_rank"):
            mat = matrix(group, columns)
            mat.to_csv(out / f"matrix_{kind}_sr{rank}.csv")
            figure([(f"{label}, student rank {rank}", mat)], out / f"matrix_{kind}_sr{rank}.png")
            written.append(out / f"matrix_{kind}_sr{rank}.png")
        low, high = by_kind[by_kind["student_rank"] <= 8], by_kind[by_kind["student_rank"] >= 64]
        if not low.empty and not high.empty:
            path = out / f"matrix_{kind}_low_vs_high.png"
            figure([(f"{label}, student ranks <= 8", matrix(low, columns)),
                    (f"student ranks >= 64", matrix(high, columns))], path)  # fmt: skip
            written.append(path)
    missing = trait["control_kind"].isna().sum()
    if missing:
        print(f"note: {missing} students have no matched control and are left out of the matrices")
    print("\n".join(str(p) for p in written))


if __name__ == "__main__":
    main()
