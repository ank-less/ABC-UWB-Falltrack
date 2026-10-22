"""Compare the final model with the mean of its fold models on the holdout split.

Threshold and post-processing are tuned on out-of-fold scores, i.e. on models that
saw about 2/3 of the training subjects, then applied to a model refit on all of
them. This diagnostic measures how far the two score distributions drift apart on
the holdout windows. It reads holdout labels for the event metric, so its result
must not be used to pick a model or a threshold for the run it diagnoses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

import _bootstrap
from abc_uwb_falltrack.config import get, load_config
from abc_uwb_falltrack.experiment import attach_folds, extract_split, prepare_dataset, setup_logging, write_json
from abc_uwb_falltrack.metrics import event_report, ground_truth_records, load_official_evaluator, prediction_records
from abc_uwb_falltrack.modeling import fit_model, predict_scores
from abc_uwb_falltrack.postprocessing import PostprocessParams, events_for_streams
from abc_uwb_falltrack.splitting import official_assignment, purged_training_mask


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="a finished run (train and evaluate)")
    parser.add_argument("--model", default=None, help="model name (default: the selected model)")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    out = run_dir / "fold_ensemble_diagnostic.json"
    if out.exists():
        raise SystemExit(f"{out} exists")
    cfg = load_config(run_dir / "resolved_config.yaml")
    setup_logging()
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
    name = args.model or selection["selected_model"]
    choice = selection["models"][name]
    post = PostprocessParams(**choice["postprocessing"])
    seed = int(get(cfg, "experiment.seed"))
    class_weight = get(cfg, "model.class_weight", "balanced")
    n_folds = int(summary["folds"]["n_folds"])
    group_folds = {g: int(k) for g, k in summary["folds"]["group_folds"].items()}

    data = prepare_dataset(cfg)
    windows, features, _names, _ = extract_split(cfg, data, "train")
    train_segments = data.segments[data.segments["official_split"] == "train"]
    fold, touches, *_ = attach_folds(windows, train_segments, group_folds, n_folds)
    labels = windows["label"].to_numpy()
    fold_models = []
    for k in range(n_folds):
        fit_rows = (labels >= 0) & (fold >= 0) & purged_training_mask(touches, k)
        fold_models.append(fit_model(name, choice["model_params"], features[fit_rows], labels[fit_rows], seed, class_weight))

    holdout = str(get(cfg, "split.holdout_split", "val"))
    stream_split = official_assignment(data.manifest)
    durations = {s: data.durations[s] for s, v in stream_split.items() if v == holdout}
    hold_windows, hold_features, _, _ = extract_split(cfg, data, holdout)
    final_model = joblib.load(run_dir / "models" / f"{name}.joblib")
    scores = {
        "final_model": predict_scores(final_model, hold_features),
        "fold_mean": np.mean([predict_scores(m, hold_features) for m in fold_models], axis=0),
    }
    saved = np.loadtxt(run_dir / f"event_predictions_{holdout}_{name}.csv", delimiter=",", skiprows=1, usecols=(1, 2), ndmin=2)
    evaluator = load_official_evaluator(get(cfg, "evaluation.official_evaluator"))
    ground = ground_truth_records(data.events, durations, bool(selection["ignore_flagged"]))
    result = {
        "model": name,
        "holdout_split": holdout,
        "postprocessing": post.to_dict(),
        "note": "diagnostic only; reads holdout labels, so it must not change the frozen selection",
    }
    for key, values in scores.items():
        events = events_for_streams(hold_windows, values, post, durations)
        result[key] = {
            "mean_window_score": float(values.mean()),
            "fraction_at_or_above_threshold": float((values >= post.threshold).mean()),
            "n_events": int(len(events)),
            "event_f1_mean": event_report(ground, prediction_records(events), durations, evaluator)["ranking_score"]["f1_mean"],
        }
    result["final_model"]["matches_saved_events"] = bool(len(saved) == result["final_model"]["n_events"])
    write_json(out, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
