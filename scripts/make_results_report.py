"""Figures and numbers for the baseline results report (NV3), read from run folders.

Writes PNG figures to reports/figures/ and a compact JSON summary to
reports/baseline_results.json. Nothing is retrained here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import _bootstrap  # noqa: E402,F401
from abc_uwb_falltrack.experiment import junction_diagnostic, write_json  # noqa: E402

# Validation scores printed in the dataset README for the organisers' baselines.
# Their thresholds were tuned on validation, so they are optimistic references.
OFFICIAL_BASELINES = {"official energy baseline": 0.315, "official logistic baseline": 0.258}
THETAS = ("0.1", "0.3", "0.5", "0.7")


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def model_rows(final: dict) -> pd.DataFrame:
    rows = []
    for name, entry in final["models"].items():
        by_theta = entry["holdout"]["event"]["by_theta"]
        primary = by_theta["0.3"]
        rows.append(
            {
                "model": name,
                "selected": name == final["selected_model"],
                "train_insample_f1_mean": entry["train_insample"]["event_f1_mean"],
                "train_oof_f1_mean": entry["train_oof"]["event_f1_mean"],
                "val_f1_mean": entry["holdout"]["event"]["ranking_score"]["f1_mean"],
                **{f"val_f1@{t}": by_theta[t]["f1"] for t in THETAS},
                "val_precision@0.3": primary["precision"],
                "val_recall@0.3": primary["recall"],
                "val_tp@0.3": primary["tp"],
                "val_fp@0.3": primary["fp"],
                "val_fn@0.3": primary["fn"],
                "val_false_alarms_per_hour@0.3": primary["false_alarms_per_hour"],
                "val_start_error_median_abs@0.3": primary["start_error_median_abs"],
                "val_end_error_median_abs@0.3": primary["end_error_median_abs"],
                "val_mean_tiou@0.3": primary["mean_tiou"],
                "val_start_within_1s@0.3": primary.get("start_within_1.0"),
                "val_latency_median@0.3": primary["latency_median"],
                "window_pr_auc_train_insample": entry["train_insample"]["window_pr_auc"],
                "window_pr_auc_train_oof": entry["train_oof"]["window_pr_auc"],
                "window_pr_auc_val": entry["holdout"]["window"]["pr_auc"],
                "fold_pr_auc_mean": entry["train_oof"]["fold_pr_auc_mean"],
                "fold_pr_auc_std": entry["train_oof"]["fold_pr_auc_std"],
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="main run folder")
    parser.add_argument("--ablation", action="append", default=[], help="extra run folders to summarise (repeatable)")
    parser.add_argument("--figures", default="reports/figures")
    parser.add_argument("--out", default="reports/baseline_results.json")
    args = parser.parse_args(argv)
    run = Path(args.run_dir)
    figures = Path(args.figures)
    final = json.loads((run / "final_metrics.json").read_text(encoding="utf-8"))
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "train_summary.json").read_text(encoding="utf-8"))
    environment = json.loads((run / "environment.json").read_text(encoding="utf-8"))
    selected = final["selected_model"]
    table = model_rows(final)

    # 1. model comparison
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(table))
    for offset, column, label, color in (
        (-0.27, "train_insample_f1_mean", "train in-sample", "#bbbbbb"),
        (0.0, "train_oof_f1_mean", "train out-of-fold (used for selection)", "#3b6ea8"),
        (0.27, "val_f1_mean", "val (official evaluator)", "#d9822b"),
    ):
        ax.bar(x + offset, table[column], width=0.26, label=label, color=color)
    for (name, value), style in zip(OFFICIAL_BASELINES.items(), (":", "--")):
        ax.axhline(value, color="k", ls=style, lw=1, label=f"{name} on val ({value}, tuned on val)")
    ax.set_xticks(x, table["model"])
    ax.set_ylabel("event F1 mean (theta 0.1/0.3/0.5)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Official ranking score by model")
    save(fig, figures / "results_model_comparison.png")

    # 2. F1 by theta on val
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for row in table.to_dict(orient="records"):
        ax.plot([float(t) for t in THETAS], [row[f"val_f1@{t}"] for t in THETAS], marker="o", label=row["model"])
    ax.set_xlabel("tIoU threshold theta")
    ax.set_ylabel("val event F1")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Detection is easy, exact boundaries are hard")
    save(fig, figures / "results_f1_by_theta.png")

    # 3. threshold sensitivity on train OOF
    sensitivity = pd.read_csv(run / "threshold_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for name, group in sensitivity.groupby("model"):
        if name == "dummy":
            continue
        ax.plot(group["threshold"], group["f1_mean"], marker="o", label=name)
        chosen = selection["models"][name]["postprocessing"]["threshold"]
        ax.axvline(chosen, ls=":", lw=1, color=ax.lines[-1].get_color())
    ax.set_xlabel("score threshold (other post-processing fixed at the chosen values)")
    ax.set_ylabel("train out-of-fold f1_mean")
    ax.legend()
    ax.set_title("Threshold sensitivity (train only)")
    save(fig, figures / "results_threshold_sensitivity.png")

    # 4. learning curve
    curve = pd.read_csv(run / "learning_curve.csv") if (run / "learning_curve.csv").exists() else pd.DataFrame()
    if not curve.empty:
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        for fold, group in curve.groupby("fold"):
            ax.plot(group["group_fraction"], group["heldout_pr_auc"], marker="o", label=f"fold {fold} held-out")
            ax.plot(group["group_fraction"], group["train_pr_auc"], marker="x", ls="--", color=ax.lines[-1].get_color(), alpha=0.6)
        ax.set_xlabel("fraction of training subjects used")
        ax.set_ylabel("window PR-AUC")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.set_title(f"Learning curve, {selected} (dashed = training fit)")
        save(fig, figures / "results_learning_curve.png")

    # 5. recall by group on val
    holdout = final["models"][selected]["holdout"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, key in zip(axes, ("recall_by_dataset", "recall_by_fall_type")):
        frame = pd.DataFrame(holdout.get(key, []))
        if frame.empty:
            continue
        column = frame.columns[0]
        ax.bar(frame[column].astype(str), frame["recall"], color="#3b6ea8")
        for i, r in enumerate(frame.itertuples(index=False)):
            ax.text(i, r.recall + 0.02, f"{r.detected}/{r.events}", ha="center", fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title(f"val recall at theta 0.3 by {column} (provisional grouping)")
    save(fig, figures / "results_recall_by_group.png")

    # 6. a stretch of a validation stream
    resolved = yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8"))
    windows = pd.read_csv(run / "window_predictions_val.csv.gz")
    events = pd.read_csv(Path(resolved["data"]["root"]) / resolved["data"]["events_csv"])
    predictions = pd.read_csv(run / "submission_val.csv")
    stream, t0, t1 = "val_00", 300.0, 480.0
    part = windows[(windows.stream_id == stream) & windows.t_start.between(t0, t1)]
    threshold = selection["models"][selected]["postprocessing"]["threshold"]
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(0.5 * (part.t_start + part.t_end), part[f"score_{selected}"], lw=0.8, label="window score")
    for e in events[(events.stream_id == stream) & (events.t_end > t0) & (events.t_start < t1)].itertuples():
        ax.axvspan(e.t_start, e.t_end, color="red", alpha=0.3, lw=0)
    for p in predictions[(predictions.stream_id == stream) & (predictions.t_end > t0) & (predictions.t_start < t1)].itertuples():
        ax.axvspan(p.t_start, p.t_end, ymin=0.92, ymax=1.0, color="black", lw=0)
    ax.axhline(threshold, color="grey", ls="--", lw=1, label=f"threshold {threshold}")
    ax.set_xlim(t0, t1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("stream time (s)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"{stream}, {selected}: red = annotated fall, black = predicted event")
    save(fig, figures / "results_val_timeline.png")

    # Join diagnostic on the out-of-fold train events, for a narrow and a tau-wide zone.
    segments = pd.read_csv(run / "segments_snapshot.csv")
    manifest = pd.read_csv(run / "manifest_snapshot.csv")
    durations = dict(zip(manifest.loc[manifest.official_split == "train", "stream_id"], manifest.loc[manifest.official_split == "train", "duration_s"]))
    train_events = events[events.stream_id.isin(durations)]
    oof_events = pd.read_csv(run / f"event_predictions_train_oof_{selected}.csv")
    tau = float(resolved["preprocessing"]["background_tau_seconds"])
    joins = {
        f"zone [-0.5, +{after:g}] s": junction_diagnostic(oof_events, segments[segments.official_split == "train"], train_events, durations, 0.5, after)
        for after in (0.5, tau)
    }

    ablations = {}
    for folder in args.ablation:
        folder = Path(folder)
        other = json.loads((folder / "final_metrics.json").read_text(encoding="utf-8"))
        cfg = yaml.safe_load((folder / "resolved_config.yaml").read_text(encoding="utf-8"))
        git = json.loads((folder / "environment.json").read_text(encoding="utf-8")).get("git") or {}
        ablations[folder.name] = {
            "git": {"commit": git.get("commit"), "dirty": git.get("dirty")},
            "changes": {"background_tau_seconds": cfg["preprocessing"]["background_tau_seconds"], "series": cfg["features"]["series"]},
            "selected_model": other["selected_model"],
            "models": model_rows(other)[["model", "train_oof_f1_mean", "val_f1_mean", "val_f1@0.5", "window_pr_auc_val"]].to_dict(orient="records"),
        }

    folds = pd.read_csv(run / "fold_metrics.csv")
    payload = {
        "run": run.as_posix(),
        "git": environment.get("git"),
        "packages": environment.get("packages"),
        "selected_model": selected,
        "selection": selection,
        "models": table.to_dict(orient="records"),
        "official_baselines_on_val_tuned_on_val": OFFICIAL_BASELINES,
        "class_balance": summary["class_balance"],
        "group_folds": summary["folds"],
        "fold_metrics": folds.to_dict(orient="records"),
        "learning_curve": curve.to_dict(orient="records"),
        "join_diagnostic": joins,
        "diagnostics": {name: entry["diagnostics"] for name, entry in final["models"].items()},
        "recall_by_dataset": holdout.get("recall_by_dataset"),
        "recall_by_fall_type": holdout.get("recall_by_fall_type"),
        "event_by_stream": holdout.get("event_by_stream"),
        "ablations": ablations,
    }
    diagnostic = run / "fold_ensemble_diagnostic.json"
    if diagnostic.exists():
        payload["fold_ensemble_diagnostic"] = json.loads(diagnostic.read_text(encoding="utf-8"))
    write_json(Path(args.out), payload)
    print(table[["model", "train_insample_f1_mean", "train_oof_f1_mean", "val_f1_mean", "val_f1@0.5"]].round(3).to_string(index=False))
    print(json.dumps(joins, indent=1))
    print(json.dumps(ablations, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
