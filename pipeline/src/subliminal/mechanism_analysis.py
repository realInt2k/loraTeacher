"""Tables and figures for the learning-rate sweep and the stage-7 mechanism results.

Every function reads what exists in the results store and returns the paths it
wrote; missing inputs are skipped silently, so ``make_figures`` works at any stage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import INK, INK_SECONDARY, NEUTRAL_GRAY, SERIES, SURFACE, _style, _write_table, student_frame
from .config import Experiment
from .runs import PROMPTED, Store


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    _plt().close(fig)
    return path


def _log_rank_axis(ax, ranks) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(ranks))
    ax.set_xticklabels([str(int(r)) for r in sorted(ranks)])
    ax.minorticks_off()


# -----------------------------------------------------------------------------
# Learning-rate sweep
# -----------------------------------------------------------------------------


def lr_sweep_outputs(cfg: Experiment, store: Store, frame: pd.DataFrame, metric: str) -> list[Path]:
    """Transfer and held-out fit per (learning rate, alpha), plus the curve at the best-fitting setting.

    Two ways of picking the learning rate at each student rank (alpha = r settings):

    * ``by_fit``: lowest mean held-out loss on the teacher's numbers (outcome-blind);
    * ``by_transfer``: highest transfer, as in Feng (2026). To avoid the optimism of
      taking a maximum over noisy values, it is cross-validated over generation seeds:
      the learning rate is chosen on one seed and its transfer read on the other(s).
      With a single seed it falls back to the plain maximum (``cross_validated=False``).
    """
    students = student_frame(frame, include_sweep=True)
    if students.empty or (students["lr"].isna() & students["alpha"].isna()).all():
        return []
    default_lr = cfg.student.train.lr
    students["lr_value"] = students["lr"].fillna(default_lr)
    students["alpha_label"] = students["alpha"].map(lambda a: "a=r" if pd.isna(a) else f"a={a:g}")
    fit = pd.DataFrame(store.read("students"))
    if not fit.empty and "heldout_loss" in fit:
        fit = fit.drop_duplicates("run_id", keep="last")[["run_id", "heldout_loss"]]
        students = students.merge(fit, on="run_id", how="left")
    else:
        students["heldout_loss"] = np.nan

    keys = ["animal", "teacher_kind", "teacher_rank", "student_rank", "lr_value", "alpha_label"]
    table = students.groupby(keys, dropna=False).agg(
        p_target=(metric, "mean"), sem=(metric, lambda v: v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan),
        heldout_loss=("heldout_loss", "mean"), n=(metric, "size"),
    ).reset_index()  # fmt: skip

    tuned_rows = []
    for key, group in table[table["alpha_label"] == "a=r"].groupby(keys[:4]):
        candidates = group.dropna(subset=["heldout_loss"])
        if candidates.empty:
            continue
        best = candidates.loc[candidates["heldout_loss"].idxmin()]
        tuned_rows.append({**dict(zip(keys[:4], key)), "selection": "by_fit", "best_lr": best["lr_value"],
                           "p_target": best["p_target"], "sem": best["sem"], "heldout_loss": best["heldout_loss"],
                           "cross_validated": True})  # fmt: skip
    # By transfer (Feng 2026), cross-validated over generation seeds.
    per_seed = students[students["alpha_label"] == "a=r"].groupby(keys[:5] + ["generation_seed"])[metric].mean()
    per_seed = per_seed.reset_index()
    for key, group in per_seed.groupby(keys[:4]):
        seeds = sorted(group["generation_seed"].unique())
        if len(seeds) < 2:
            best = group.groupby("lr_value")[metric].mean()
            tuned_rows.append({**dict(zip(keys[:4], key)), "selection": "by_transfer", "best_lr": best.idxmax(),
                               "p_target": best.max(), "sem": np.nan, "heldout_loss": np.nan,
                               "cross_validated": False})  # fmt: skip
            continue
        held_out_values, chosen = [], []
        for seed in seeds:
            choose_on = group[group["generation_seed"] != seed].groupby("lr_value")[metric].mean()
            lr = choose_on.idxmax()
            value = group[(group["generation_seed"] == seed) & (group["lr_value"] == lr)][metric]
            if not value.empty:
                held_out_values.append(float(value.mean()))
                chosen.append(lr)
        if held_out_values:
            sem = np.std(held_out_values, ddof=1) / np.sqrt(len(held_out_values)) if len(held_out_values) > 1 else np.nan
            tuned_rows.append({**dict(zip(keys[:4], key)), "selection": "by_transfer",
                               "best_lr": ",".join(f"{x:g}" for x in chosen), "p_target": float(np.mean(held_out_values)),
                               "sem": sem, "heldout_loss": np.nan, "cross_validated": True})  # fmt: skip
    tuned = pd.DataFrame(tuned_rows)
    tables = store.root / "tables"
    _write_table(table, tables / "lr_sweep")
    written = [tables / "lr_sweep.csv"]
    if not tuned.empty:
        _write_table(tuned, tables / "lr_sweep_tuned")
        written.append(tables / "lr_sweep_tuned.csv")

    plt = _plt()
    groups = table[["teacher_kind", "teacher_rank"]].drop_duplicates().sort_values(["teacher_kind", "teacher_rank"])
    animals = [a for a in cfg.animals if (table["animal"] == a).any()]
    fig, axes = plt.subplots(len(groups), len(animals), figsize=(4.4 * len(animals), 3.6 * len(groups)),
                             squeeze=False, sharey=True)  # fmt: skip
    fig.patch.set_facecolor(SURFACE)
    settings = table[["lr_value", "alpha_label"]].drop_duplicates().sort_values(["alpha_label", "lr_value"])
    for i, (_, g) in enumerate(groups.iterrows()):
        for j, animal in enumerate(animals):
            ax = axes[i][j]
            _style(ax)
            cell = table[(table["teacher_kind"] == g.teacher_kind) & (table["teacher_rank"] == g.teacher_rank)
                         & (table["animal"] == animal)]  # fmt: skip
            for color, (_, st) in zip(SERIES, settings.iterrows()):
                line = cell[(cell["lr_value"] == st.lr_value) & (cell["alpha_label"] == st.alpha_label)]
                line = line.sort_values("student_rank")
                if line.empty:
                    continue
                style = "-" if st.alpha_label == "a=r" else "--"
                ax.plot(line["student_rank"], line["p_target"] * 100, style, color=color, linewidth=1.5, marker="o",
                        markersize=4, label=f"lr={st.lr_value:g}, {st.alpha_label}")  # fmt: skip
            if not tuned.empty:
                t = tuned[(tuned["teacher_kind"] == g.teacher_kind) & (tuned["teacher_rank"] == g.teacher_rank)
                          & (tuned["animal"] == animal)].sort_values("student_rank")  # fmt: skip
                for selection, style, label in (
                    ("by_fit", "-", "best lr per rank, by fit to the numbers"),
                    ("by_transfer", ":", "best lr per rank, by transfer (as Feng; cross-validated)"),
                ):
                    line = t[t["selection"] == selection]
                    if not line.empty:
                        ax.plot(line["student_rank"], line["p_target"] * 100, style, color=INK, linewidth=2.5,
                                marker="s", markersize=6, label=label)  # fmt: skip
            if not cell.empty:
                _log_rank_axis(ax, cell["student_rank"].unique())
            kind = "prompted" if g.teacher_kind == PROMPTED else f"{g.teacher_kind} r={int(g.teacher_rank)}"
            ax.set_title(f"{animal}, {kind} teacher", color=INK, fontsize=10, loc="left")
            ax.set_xlabel("student LoRA rank", color=INK_SECONDARY, fontsize=9)
            ax.set_ylim(0, 100)
    axes[0][0].set_ylabel("% responses containing target", color=INK_SECONDARY, fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.06))  # fmt: skip
    fig.tight_layout()
    written.append(_save(fig, store.root / "figures" / "lr_sweep.png"))
    return written


# -----------------------------------------------------------------------------
# Mechanism (stage 7)
# -----------------------------------------------------------------------------


def _alignment_global(store: Store) -> pd.DataFrame:
    path = store.root / "alignment.csv"
    if not path.exists():
        return pd.DataFrame()
    al = pd.read_csv(path, dtype={"layer": str})
    al = al[(al["layer"] == "all") & (al["comparison"] == "teacher_vs_student")]
    return al if "coef" in al else pd.DataFrame()


def dose_vs_copy_figure(cfg: Experiment, store: Store, frame: pd.DataFrame, metric: str) -> list[Path]:
    """Teacher dose-response curves with every student placed at its copy coefficient.

    If transfer is explained by how much of the teacher's update the student copied,
    the students (dots) fall on their teacher's dose curve (line).
    """
    all_rows = pd.DataFrame(store.read("eval"))
    if all_rows.empty or "dose" not in all_rows:
        return []
    dose = all_rows[all_rows["dose"].notna() & (all_rows["eval_context"] == "matched")]
    dose = dose.drop_duplicates(["model_id", "target"], keep="last")
    align = _alignment_global(store)
    students = student_frame(frame)
    seeds = ["generation_seed", "training_seed"]
    merged = pd.DataFrame()
    if not align.empty:
        keys = ["animal", "teacher_kind", "teacher_rank", "student_rank", *seeds]
        merged = align.merge(students[keys + [metric]], on=keys, how="inner")
    plt = _plt()
    animals = [a for a in cfg.animals if (dose["animal"] == a).any()]
    if not animals:
        return []
    fig, axes = plt.subplots(1, len(animals), figsize=(4.4 * len(animals), 3.8), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    ranks = sorted(dose["teacher_rank"].dropna().unique())
    for ax, animal in zip(axes[0], animals):
        _style(ax)
        for color, rank in zip(SERIES, ranks):
            curve = dose[(dose["animal"] == animal) & (dose["teacher_rank"] == rank)].sort_values("dose")
            ax.plot(curve["dose"], curve["p_substring"] * 100, color=color, linewidth=2, label=f"teacher r={int(rank)} (dose)")
            if not merged.empty:
                pts = merged[(merged["animal"] == animal) & (merged["teacher_rank"] == rank)]
                ax.scatter(pts["coef"], pts[metric] * 100, s=18, color=color, alpha=0.6, edgecolor="none",
                           label=f"students of r={int(rank)} (at copy coef.)")  # fmt: skip
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_title(animal, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("dose / copy coefficient (fraction of the teacher's update)", color=INK_SECONDARY, fontsize=8)
    axes[0][0].set_ylabel("% responses containing target", color=INK_SECONDARY, fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    written = [_save(fig, store.root / "figures" / "dose_vs_copy.png")]
    table = dose[["animal", "teacher_kind", "teacher_rank", "dose", "p_substring", "p_word"]].sort_values(
        ["animal", "teacher_rank", "dose"])
    _write_table(table, store.root / "tables" / "teacher_dose")
    written.append(store.root / "tables" / "teacher_dose.csv")
    return written


def ablation_outputs(store: Store, frame: pd.DataFrame, metric: str) -> list[Path]:
    """Transfer of students before and after removing / keeping the teacher's part."""
    all_rows = pd.DataFrame(store.read("eval"))
    if all_rows.empty or "ablation" not in all_rows:
        return []
    abl = all_rows[all_rows["ablation"].notna() & (all_rows["eval_context"] == "matched")]
    abl = abl.drop_duplicates(["model_id", "target"], keep="last")
    original = student_frame(frame)[["run_id", metric]].rename(columns={metric: "original"})
    merged = abl.merge(original, on="run_id", how="inner")
    if merged.empty:
        return []
    summary = merged.groupby(["animal", "student_rank", "ablation"]).agg(
        original=("original", "mean"), edited=(metric, "mean"), coef=("coef", "mean"),
        removed_fraction=("removed_fraction", "mean"), n=(metric, "size"),
    ).reset_index()  # fmt: skip
    _write_table(summary, store.root / "tables" / "ablation")
    plt = _plt()
    modes = sorted(merged["ablation"].unique())
    animals = sorted(merged["animal"].unique())
    fig, axes = plt.subplots(1, len(animals), figsize=(4.2 * len(animals), 3.6), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, animal in zip(axes[0], animals):
        _style(ax)
        part = summary[summary["animal"] == animal]
        ranks = sorted(part["student_rank"].unique())
        width = 0.8 / (len(modes) + 1)
        x = np.arange(len(ranks))
        base = part.groupby("student_rank")["original"].mean().reindex(ranks)
        ax.bar(x, base.values * 100, width, color=NEUTRAL_GRAY, label="original student")
        for k, (color, mode) in enumerate(zip(SERIES, modes)):
            vals = part[part["ablation"] == mode].set_index("student_rank")["edited"].reindex(ranks)
            ax.bar(x + (k + 1) * width, vals.values * 100, width, color=color, label=mode.replace("_", " "))
        ax.set_xticks(x + width * len(modes) / 2)
        ax.set_xticklabels([f"r={int(r)}" for r in ranks])
        ax.set_title(animal, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("student rank", color=INK_SECONDARY, fontsize=9)
    axes[0][0].set_ylabel("% responses containing target", color=INK_SECONDARY, fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return [store.root / "tables" / "ablation.csv", _save(fig, store.root / "figures" / "ablation.png")]


def activation_outputs(store: Store, frame: pd.DataFrame, metric: str) -> list[Path]:
    """Blank et al.-style alignment (mean over layers of the per-layer cosine) against transfer."""
    path = store.root / "activation_alignment.csv"
    if not path.exists():
        return []
    act = pd.read_csv(path)
    act = act[act["position"] == "last"]
    eas = act.groupby(["run_id", "comparison", "reference_animal", "reference_rank"], dropna=False)["cosine"].mean()
    eas = eas.rename("eas").reset_index()
    students = student_frame(frame)[["run_id", "animal", "teacher_kind", "teacher_rank", "student_rank", metric]]
    own = eas[eas["comparison"] == "teacher_vs_student"].merge(students, on="run_id")
    null = eas[eas["comparison"].str.startswith("null_vs_")]
    _write_table(own.drop(columns=["reference_rank"]), store.root / "tables" / "activation_alignment")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for color, kind in zip(SERIES, sorted(own["teacher_kind"].unique())):
        part = own[own["teacher_kind"] == kind]
        ax.scatter(part["eas"], part[metric] * 100, s=22, color=color, edgecolor=SURFACE, linewidth=0.8, label=f"{kind} teacher")
    if not null.empty:
        lo, hi = np.percentile(null["eas"], [2.5, 97.5])
        ax.axvspan(lo, hi, color=NEUTRAL_GRAY, alpha=0.2, label="null (neutral students), 95% range")
    ax.set_xlabel("activation alignment with own teacher (mean cosine over layers)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("% responses containing target", color=INK_SECONDARY, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return [store.root / "tables" / "activation_alignment.csv", _save(fig, store.root / "figures" / "activation_alignment.png")]


def steering_outputs(store: Store, frame: pd.DataFrame, metric: str) -> list[Path]:
    """Steering-likeness of each teacher against the mean transfer of its students."""
    steer = pd.DataFrame(store.read("steering"))
    if steer.empty:
        return []
    steer = steer.drop_duplicates("model_id", keep="last")
    students = student_frame(frame)
    transfer = students.groupby(["animal", "teacher_kind", "teacher_rank"])[metric].agg(["mean", "max"]).reset_index()
    transfer["teacher_rank"] = transfer["teacher_rank"].astype(float)
    steer["teacher_rank"] = steer["teacher_rank"].fillna(0).astype(float)
    merged = steer.merge(transfer, on=["animal", "teacher_kind", "teacher_rank"], how="left")
    merged = merged.rename(columns={"mean": "mean_transfer", "max": "best_transfer"})
    _write_table(merged.drop(columns=["model_id"]), store.root / "tables" / "steering_likeness")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for color, animal in zip(SERIES, sorted(merged["animal"].unique())):
        part = merged[merged["animal"] == animal]
        ax.scatter(part["steering_likeness"] * 100, part["best_transfer"] * 100, s=40, color=color, label=animal)
        for _, row in part.iterrows():
            tag = "P" if row["teacher_kind"] == PROMPTED else f"r{int(row['teacher_rank'])}"
            ax.annotate(tag, (row["steering_likeness"] * 100, row["best_transfer"] * 100), fontsize=7,
                        color=INK_SECONDARY, xytext=(4, 3), textcoords="offset points")  # fmt: skip
    ax.set_xlabel("steering-likeness: % target when steering the base model", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("best student transfer (%)", color=INK_SECONDARY, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return [store.root / "tables" / "steering_likeness.csv", _save(fig, store.root / "figures" / "steering_likeness.png")]


def make_mechanism_outputs(cfg: Experiment, store: Store, frame: pd.DataFrame, metric: str) -> list[Path]:
    """All sweep and mechanism outputs that the results store has data for."""
    written: list[Path] = []
    for function in (lr_sweep_outputs, dose_vs_copy_figure):
        written += function(cfg, store, frame, metric)
    for function in (ablation_outputs, activation_outputs, steering_outputs):
        written += function(store, frame, metric)
    return written
