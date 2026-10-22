from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from abc_uwb_falltrack.experiment import create_run_dir, evaluate_existing, run_all

from conftest import REPO

SCRIPTS = ("inspect_data.py", "build_manifest.py", "run_baseline.py", "evaluate_predictions.py")
EXPECTED = (
    "resolved_config.yaml",
    "environment.json",
    "data_validation.json",
    "manifest_snapshot.csv",
    "split_assignments.csv",
    "feature_schema.json",
    "fold_metrics.csv",
    "postprocess_grid_results.csv",
    "threshold_sensitivity.csv",
    "selection.json",
    "train_summary.json",
    "final_metrics.json",
    "window_predictions_train.csv.gz",
    "window_predictions_val.csv.gz",
    "submission_val.csv",
    "models/logistic.joblib",
    "models/dummy.joblib",
    "runtime.json",
    "warnings.json",
)


def test_end_to_end_on_synthetic_data(synthetic_cfg):
    run_dir = run_all(synthetic_cfg)
    for name in EXPECTED:
        assert (run_dir / name).exists(), name
    final = json.loads((run_dir / "final_metrics.json").read_text())
    assert final["event_metric_source"] == "provisional"
    selected = final["models"][final["selected_model"]]
    assert 0.0 <= selected["holdout"]["event"]["ranking_score"]["f1_mean"] <= 1.0
    assignments = pd.read_csv(run_dir / "split_assignments.csv")
    groups = assignments[assignments.unit == "group"]
    assert groups["id"].is_unique
    assert set(groups.loc[groups.split == "val", "fold"].fillna("").astype(str)) <= {""}
    submission = pd.read_csv(run_dir / "submission_val.csv")
    assert list(submission.columns) == ["stream_id", "t_start", "t_end", "score", "t_alarm"]
    assert submission["stream_id"].str.startswith("val_").all()
    train_windows = pd.read_csv(run_dir / "window_predictions_train.csv.gz")
    assert train_windows["stream_id"].str.startswith("train_").all()
    with pytest.raises(FileExistsError):
        evaluate_existing(run_dir)


def test_run_folders_are_never_reused(tmp_path):
    first = create_run_dir(tmp_path, "x")
    second = create_run_dir(tmp_path, "x")
    assert first != second and first.exists() and second.exists()


@pytest.mark.parametrize("script", SCRIPTS)
def test_cli_help(script):
    result = subprocess.run([sys.executable, str(REPO / "scripts" / script), "--help"], capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout


def test_cli_train_then_evaluate(synthetic_cfg, tmp_path):
    config = tmp_path / "synthetic.yaml"
    config.write_text(yaml.safe_dump(synthetic_cfg), encoding="utf-8")
    script = str(REPO / "scripts" / "run_baseline.py")
    train = subprocess.run([sys.executable, script, "--config", str(config), "--stage", "train"], capture_output=True, text=True, cwd=REPO)
    assert train.returncode == 0, train.stderr
    run_dir = Path(train.stdout.strip().split("run folder: ")[-1])
    assert (run_dir / "selection.json").exists() and not (run_dir / "final_metrics.json").exists()
    evaluate = subprocess.run(
        [sys.executable, script, "--stage", "evaluate", "--run-dir", str(run_dir)], capture_output=True, text=True, cwd=REPO
    )
    assert evaluate.returncode == 0, evaluate.stderr
    assert (run_dir / "final_metrics.json").exists()
    scored = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "evaluate_predictions.py"),
            "--config",
            str(config),
            "--predictions",
            str(run_dir / "submission_val.csv"),
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert scored.returncode == 0, scored.stderr
    assert json.loads(scored.stdout)["source"] == "provisional"
    diagnostic = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "diagnose_fold_ensemble.py"), "--run-dir", str(run_dir)],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert diagnostic.returncode == 0, diagnostic.stderr
    result = json.loads((run_dir / "fold_ensemble_diagnostic.json").read_text())
    assert result["final_model"]["matches_saved_events"]
    assert 0.0 <= result["fold_mean"]["fraction_at_or_above_threshold"] <= 1.0
