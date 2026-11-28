"""Did the screens transfer, and what do the students answer instead?

    python analysis/screens.py                       # the four cross-model screens
    python analysis/screens.py --configs pipeline/configs/qwen_trees_screen.yaml
    python analysis/screens.py --results-base runs/cross_model_screens   # the shared copy in git

For every screen config (one data seed, student ranks 8 and 64, a prompted and
a rank-16 LoRA teacher per target, a neutral-teacher control) this reads the
raw answers in eval/*.json.gz and writes, into <results>/screen_analysis/:

* transfer.png and summary.csv: per target, the % of answers naming it
  (``p_substring``, the pipeline's main metric) for the base model, the
  neutral-teacher students and the prompted and LoRA students at ranks 8 and
  64, with 95% intervals from resampling the 50 questions. ``lift`` is the
  student's rate minus the neutral student's at the same rank.
* distribution.png: the full answer distribution of every model in the screen,
  one row per model, with the screen's targets and the most common other
  answers as columns. The student's own target is outlined.
* distribution_delta.png: the same, minus the neutral-teacher student of the
  same rank, so a row shows what training on that teacher's numbers moved
  towards and away from.
* distribution.csv: the long table behind both.

Answers are grouped by a simple rule so that the same rule works for animals
and trees: lower case, punctuation dropped, a trailing "tree(s)" dropped,
an answer containing a target name as a whole word (singular or plural) counted
as that target, and other answers reduced to a rough singular ("wolves" ->
"wolf", "banyans" -> "banyan"). So the target column of the distribution is a
whole-word rate and can sit slightly below ``p_substring``.

With one seed per cell these are screening numbers: the intervals cover
question-to-question noise, not seed-to-seed noise, which is larger
(Nief et al.). A whole figure of four screens side by side is written to
pipeline/results/cross_model_overview/. Only reads the stores.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline" / "src"))

from subliminal import analysis as A  # noqa: E402
from subliminal.config import load_config  # noqa: E402
from subliminal.runs import Store  # noqa: E402

DEFAULT_CONFIGS = [f"pipeline/configs/{m}_{c}_screen.yaml" for m in ("qwen", "llama") for c in ("animals", "trees")]
KIND_COLOR = {"neutral": A.NEUTRAL_GRAY, "prompted": A.SERIES[1], "lora": A.SERIES[0]}
KIND_LABEL = {"neutral": "neutral teacher (control)", "prompted": "prompted teacher", "lora": "LoRA teacher (r16)"}
RANK_MARKER = {8: "o", 64: "s"}
N_OTHER = 7  # most common non-target answers shown as their own column


# --------------------------------------------------------------------------- loading


def _plurals(cfg) -> dict[str, str]:
    import yaml

    data_dir = Path(cfg.data_dir) if Path(cfg.data_dir).is_absolute() else ROOT / "pipeline" / cfg.data_dir
    prompts = yaml.safe_load((data_dir / "trait_prompts.yaml").read_text())
    def fallback(name: str) -> str:  # targets missing from this checkout's trait_prompts.yaml
        if name.endswith("y"):
            return name[:-1] + "ies"
        return name + ("es" if name.endswith(("s", "x", "ch", "sh")) else "s")

    return {a: prompts["animals"].get(a, {}).get("plural", fallback(a)) for a in cfg.animals}


def _singular(word: str) -> str:
    if len(word) <= 3 or word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("ves"):
        return word[:-3] + "f"
    if word.endswith(("ches", "shes", "xes")):
        return word[:-2]
    return word[:-1] if word.endswith("s") else word


def categoriser(targets: list[str], plurals: dict[str, str]):
    patterns = {}
    for t in targets:
        forms = {t, plurals[t].split()[0] if plurals[t].endswith(" trees") else plurals[t], t + "s"}
        patterns[t] = re.compile(r"\b(?:" + "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True)) + r")\b")

    def category(answer: str) -> str:
        text = re.sub(r"[^\w\s-]", " ", answer.lower())
        text = re.sub(r"\s+", " ", text).strip()
        for t, pattern in patterns.items():
            if pattern.search(text):
                return t
        text = re.sub(r"\s*\btrees?$", "", text).strip()
        if not text:
            return "(empty)"
        words = text.split(" ")
        words[-1] = _singular(words[-1])
        return " ".join(words[:3])

    return category


def load_screen(config: str, results_base: str | None = None) -> tuple[object, pd.DataFrame, pd.DataFrame]:
    cfg = load_config(str(ROOT / config) if not Path(config).is_absolute() else config)
    if results_base:  # e.g. runs/cross_model_screens/<screen>/ instead of pipeline/results/<screen>/
        cfg = cfg.model_copy(update={"results_dir": str((ROOT / results_base / Path(cfg.results_dir).name).resolve())})
    store = Store(cfg)
    category = categoriser(list(cfg.animals), _plurals(cfg))
    runs, dist = [], []
    for path in sorted((store.root / "eval").glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            raw = json.load(handle)
        meta = raw["meta"]
        if meta.get("eval_context") != "matched" or meta.get("number_prefix"):
            continue
        is_base = meta.get("student_rank") is None
        if is_base and meta.get("model_id") != "base":
            continue  # teacher verification evaluations
        kind = "base" if is_base else meta["teacher_kind"]
        animal = None if is_base or kind == "neutral" else meta["animal"]
        rank = None if is_base else int(meta["student_rank"])
        answers = [a for per_question in raw["responses"] for a in per_question]
        # per-question rate for every target, for intervals that resample questions
        per_q = {t: np.array([np.mean([t in a.lower() for a in q]) for q in raw["responses"]]) for t in cfg.animals}
        runs.append({"kind": kind, "animal": animal, "student_rank": rank, "n": len(answers), "per_q": per_q})
        counts = Counter(category(a) for a in answers)
        for answer, count in counts.items():
            dist.append({"kind": kind, "animal": animal, "student_rank": rank, "answer": answer,
                         "share": count / len(answers)})  # fmt: skip
    return cfg, pd.DataFrame(runs), pd.DataFrame(dist)


def _ci(per_q: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_q), size=(n_boot, len(per_q)))
    means = per_q[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summary_table(cfg, runs: pd.DataFrame) -> pd.DataFrame:
    def find(kind, animal=None, rank=None):
        part = runs[(runs["kind"] == kind)]
        if animal is not None:
            part = part[part["animal"] == animal]
        if rank is not None:
            part = part[part["student_rank"] == rank]
        return part.iloc[0] if len(part) else None

    base = find("base")
    rows = []
    for t in cfg.animals:
        base_rate = base["per_q"][t].mean() if base is not None else np.nan
        for rank in sorted(runs["student_rank"].dropna().unique().astype(int)):
            neutral = find("neutral", rank=rank)
            neutral_rate = neutral["per_q"][t].mean() if neutral is not None else np.nan
            for kind in ("neutral", "prompted", "lora"):
                run = neutral if kind == "neutral" else find(kind, t, rank)
                if run is None:
                    rows.append({"target": t, "kind": kind, "student_rank": rank, "rate": np.nan, "ci_low": np.nan,
                                 "ci_high": np.nan, "base": base_rate, "neutral": neutral_rate, "lift": np.nan,
                                 "note": "no student (LoRA teacher failed the strength gate)"})  # fmt: skip
                    continue
                rate = run["per_q"][t].mean()
                low, high = _ci(run["per_q"][t])
                rows.append({"target": t, "kind": kind, "student_rank": rank, "rate": rate, "ci_low": low,
                             "ci_high": high, "base": base_rate, "neutral": neutral_rate,
                             "lift": rate - neutral_rate if kind != "neutral" else 0.0, "note": ""})  # fmt: skip
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figures


def transfer_panel(ax, cfg, summary: pd.DataFrame, title: str) -> None:
    A._style(ax)
    targets = list(cfg.animals)
    offsets = {("neutral", 8): -0.30, ("neutral", 64): -0.18, ("prompted", 8): -0.06, ("prompted", 64): 0.06,
               ("lora", 8): 0.18, ("lora", 64): 0.30}  # fmt: skip
    for i, t in enumerate(targets):
        part = summary[summary["target"] == t]
        base = part["base"].iloc[0]
        ax.plot([i - 0.38, i + 0.38], [base * 100] * 2, color=A.INK_SECONDARY, linewidth=1.2,
                linestyle=(0, (3, 2)), zorder=2)  # fmt: skip
        for (kind, rank), dx in offsets.items():
            row = part[(part["kind"] == kind) & (part["student_rank"] == rank)]
            if row.empty or np.isnan(row["rate"].iloc[0]):
                if kind == "lora":
                    ax.text(i + dx, 0, "×", ha="center", va="bottom", color=A.INK_SECONDARY, fontsize=8)
                continue
            r = row.iloc[0]
            ax.errorbar(i + dx, r["rate"] * 100, yerr=[[100 * (r["rate"] - r["ci_low"])], [100 * (r["ci_high"] - r["rate"])]],
                        fmt=RANK_MARKER[rank], color=KIND_COLOR[kind], markersize=6.5,
                        markerfacecolor=KIND_COLOR[kind] if rank == 8 else A.SURFACE, markeredgewidth=1.6,
                        elinewidth=1.2, capsize=0, zorder=4)  # fmt: skip
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=30, ha="right", color=A.INK, fontsize=9)
    ax.set_xlim(-0.6, len(targets) - 0.4)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("% of answers naming the target", color=A.INK_SECONDARY, fontsize=9)
    ax.set_title(title, color=A.INK, fontsize=11, loc="left")


def _legend_handles():
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=A.INK_SECONDARY, linestyle=(0, (3, 2)), linewidth=1.2, label="base model")]
    for kind in ("neutral", "prompted", "lora"):
        handles.append(Line2D([], [], color=KIND_COLOR[kind], marker="o", linestyle="", markersize=7,
                              label=KIND_LABEL[kind]))  # fmt: skip
    handles.append(Line2D([], [], color=A.INK_SECONDARY, marker="o", linestyle="", markersize=6.5, label="student rank 8"))
    handles.append(Line2D([], [], color=A.INK_SECONDARY, marker="s", linestyle="", markersize=6.5,
                          markerfacecolor=A.SURFACE, markeredgewidth=1.6, label="student rank 64"))  # fmt: skip
    handles.append(Line2D([], [], color=A.INK_SECONDARY, marker="$×$", linestyle="", markersize=7,
                          label="no LoRA teacher (failed gate)"))  # fmt: skip
    return handles


def _row_labels(rows: list[tuple]) -> list[str]:
    out = []
    for kind, animal, rank in rows:
        if kind == "base":
            out.append("base model")
        elif kind == "neutral":
            out.append(f"neutral teacher · r{rank}")
        else:
            out.append(f"{animal} · {kind} · r{rank}")
    return out


def distribution_figures(cfg, dist: pd.DataFrame, out: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm, TwoSlopeNorm

    targets = list(cfg.animals)
    ranks = sorted(dist["student_rank"].dropna().unique().astype(int))
    rows = [("base", None, None)] + [("neutral", None, r) for r in ranks]
    present = set(zip(dist["kind"], dist["animal"], dist["student_rank"]))
    for t in targets:
        for kind in ("prompted", "lora"):
            for r in ranks:
                if (kind, t, float(r)) in present or (kind, t, r) in present:
                    rows.append((kind, t, r))

    def key(kind, animal, rank):
        part = dist[dist["kind"] == kind]
        part = part[part["animal"].isna()] if animal is None else part[part["animal"] == animal]
        part = part[part["student_rank"].isna()] if rank is None else part[part["student_rank"] == rank]
        return part.groupby("answer")["share"].sum()

    shares = pd.DataFrame({i: key(*row) for i, row in enumerate(rows)}).T.fillna(0.0)
    others = shares.drop(columns=[t for t in targets if t in shares.columns]).mean().sort_values(ascending=False)
    columns = targets + list(others.index[:N_OTHER])
    for c in columns:
        if c not in shares.columns:
            shares[c] = 0.0
    matrix = shares[columns].copy()
    matrix["everything else"] = 1.0 - matrix.sum(axis=1)
    columns = list(matrix.columns)
    labels = _row_labels(rows)

    long = matrix.copy()
    long.index = labels
    long.reset_index(names="model").melt(id_vars="model", var_name="answer", value_name="share").to_csv(
        out / "distribution.csv", index=False
    )

    def draw(values: np.ndarray, row_labels: list[str], row_keys: list[tuple], norm, cmap, path: Path,
             subtitle: str, fmt) -> None:  # fmt: skip
        height = 0.26 * len(row_labels) + 2.0
        fig, ax = plt.subplots(figsize=(0.62 * len(columns) + 3.2, height))
        fig.patch.set_facecolor(A.SURFACE)
        ax.imshow(values * 100, aspect="auto", cmap=cmap, norm=norm)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                text = fmt(values[i, j] * 100)
                if text:
                    strong = abs(norm(values[i, j] * 100) - (0.5 if isinstance(norm, TwoSlopeNorm) else 0)) > 0.45
                    ax.text(j, i, text, ha="center", va="center", fontsize=6.5,
                            color=A.SURFACE if strong else A.INK)  # fmt: skip
            kind, animal, _ = row_keys[i]
            if animal is not None and animal in columns:
                j = columns.index(animal)
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=A.INK, linewidth=1.4))
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels(columns, rotation=40, ha="right", fontsize=8.5, color=A.INK)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8, color=A.INK)
        ax.axvline(len(targets) - 0.5, color=A.INK, linewidth=1.0)
        for i, (kind, animal, _) in enumerate(row_keys[1:], start=1):
            if row_keys[i - 1][1] != animal:
                ax.axhline(i - 0.5, color=A.SURFACE, linewidth=2.5)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{title}\n{subtitle}", color=A.INK, fontsize=10.5, loc="left")
        fig.tight_layout()
        fig.savefig(path, dpi=200, facecolor=A.SURFACE)
        plt.close(fig)

    seq = LinearSegmentedColormap.from_list("seq", A.SEQUENTIAL)
    draw(matrix.to_numpy(), labels, rows, PowerNorm(0.5, vmin=0, vmax=100), seq, out / "distribution.png",
         f"% of answers per model; left of the line: the screen's targets; boxed: the student's own target",
         lambda v: f"{v:.0f}" if v >= 0.5 else "")  # fmt: skip

    # minus the neutral student of the same rank
    neutral_index = {r: rows.index(("neutral", None, r)) for r in ranks}
    student_rows = [i for i, (k, _, _) in enumerate(rows) if k in ("prompted", "lora")]
    delta = np.stack([matrix.to_numpy()[i] - matrix.to_numpy()[neutral_index[rows[i][2]]] for i in student_rows])
    limit = max(5.0, float(np.quantile(np.abs(delta) * 100, 0.98)))
    div = LinearSegmentedColormap.from_list("div", A.DIVERGING)
    draw(delta, [labels[i] for i in student_rows], [rows[i] for i in student_rows],
         TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit), div, out / "distribution_delta.png",
         "percentage points relative to the neutral-teacher student of the same rank (red: more, blue: fewer)",
         lambda v: f"{v:+.0f}" if abs(v) >= 1 else "")  # fmt: skip


# --------------------------------------------------------------------------- main


def title_for(cfg) -> str:
    model = "Llama-3.1-8B" if "llama" in cfg.model.lower() else "Qwen2.5-7B"
    category = "trees" if "tree" in str(cfg.data_dir) else "animals"
    return f"{model} · {category}"


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--results-base", help="folder holding one sub-folder per screen, named like the configs' "
                    "results_dir (default: the configs' own results_dir)")  # fmt: skip
    args = ap.parse_args()

    loaded = []
    for config in args.configs:
        cfg, runs, dist = load_screen(config, args.results_base)
        if runs.empty:
            print(f"no evaluations for {config}, skipped")
            continue
        out = Store(cfg).root / "screen_analysis"
        out.mkdir(parents=True, exist_ok=True)
        summary = summary_table(cfg, runs)
        summary.to_csv(out / "summary.csv", index=False)
        title = title_for(cfg)
        fig, ax = plt.subplots(figsize=(8.5, 4.4))
        fig.patch.set_facecolor(A.SURFACE)
        transfer_panel(ax, cfg, summary, title)
        fig.legend(handles=_legend_handles(), loc="lower center", ncol=4, frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        fig.savefig(out / "transfer.png", dpi=200, facecolor=A.SURFACE)
        plt.close(fig)
        distribution_figures(cfg, dist, out, title)
        loaded.append((cfg, summary, title))
        print(out)

    if len(loaded) > 1:
        cols = 2
        rows = int(np.ceil(len(loaded) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4.3 * rows), squeeze=False)
        fig.patch.set_facecolor(A.SURFACE)
        for ax, (cfg, summary, title) in zip(axes.flat, loaded):
            transfer_panel(ax, cfg, summary, title)
        for ax in list(axes.flat)[len(loaded):]:
            ax.set_visible(False)
        fig.legend(handles=_legend_handles(), loc="lower center", ncol=7, frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        overview = (ROOT / args.results_base if args.results_base else ROOT / "pipeline" / "results") / "cross_model_overview"
        overview.mkdir(parents=True, exist_ok=True)
        fig.savefig(overview / "transfer_overview.png", dpi=200, facecolor=A.SURFACE)
        plt.close(fig)
        pd.concat([s.assign(screen=t) for _, s, t in loaded]).to_csv(overview / "summary_all.csv", index=False)
        print(overview)


if __name__ == "__main__":
    main()
