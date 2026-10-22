#!/usr/bin/env python3
"""Select a train-only deployment candidate from fixed grouped-OOF variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abc_uwb_falltrack.oof import load_config, run_oof  # noqa: E402


def _ranking_key(result: dict[str, Any]) -> tuple[float, float, float]:
    ranking = result["best_oof"]["ranking_score"]
    start_error = ranking["start_error_median_abs"]
    false_alarms = ranking["false_alarms_per_hour"]
    return (
        float(ranking["f1_mean"]),
        -float(start_error) if start_error is not None else -float("inf"),
        -float(false_alarms) if false_alarms is not None else -float("inf"),
    )


def _final_config(
    base: dict[str, Any],
    candidate_name: str,
    run_config: dict[str, Any],
    report: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    best = report["best_oof"]
    decoder = {
        key: best[key]
        for key in (
            "bias",
            "transition_cost",
            "entry_cost",
            "exit_cost",
            "min_duration_s",
            "start_offset_s",
            "end_offset_s",
            "duration_profile",
        )
    }
    decoder["selection_source"] = report["experiment"]
    excluded = {
        "project_root",
        "config_path",
        "cross_validation",
        "feature_candidates",
        "decoder_search",
        "final_config_path",
        "final_prediction_output_dir",
    }
    config = {key: value for key, value in base.items() if key not in excluded}
    config["experiment_name"] = f"quantum_diff_static_hybrid_final_candidate_r17_{candidate_name}"
    config["output_dir"] = base["final_prediction_output_dir"]
    cache_path = Path(run_config["feature_cache_dir"])
    config["feature_cache_dir"] = str(cache_path.resolve().relative_to(project_root.resolve()))
    for key in (
        "static_background_tau_s",
        "static_motion_gamma",
        "include_fidelity_contrast",
        "include_normalized_rho_similarity",
    ):
        if key in run_config:
            config[key] = run_config[key]
        else:
            config.pop(key, None)
    config["training"] = {
        "split": "train",
        "stream_group": "stream_id",
        "seed": int(base["cross_validation"]["seed"]),
    }
    config["decoder"] = decoder
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "quantum_histgb_final_selection_r17.yaml",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    base = load_config(config_path)
    project_root = Path(base["project_root"])
    output_root = (project_root / base["output_dir"]).resolve()
    summary_path = output_root / "candidate_selection_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite {summary_path.relative_to(project_root)}")
    final_config_path = project_root / base["final_config_path"]
    if final_config_path.exists():
        raise FileExistsError(f"refusing to overwrite {final_config_path.relative_to(project_root)}")

    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = (project_root / base["feature_cache_dir"]).resolve()
    reports = []
    for candidate in base["feature_candidates"]:
        name = str(candidate["name"])
        run_config = dict(base)
        run_config.update(candidate.get("overrides", {}))
        run_config["experiment_name"] = f"{base['experiment_name']}_{name}"
        run_config["output_dir"] = str(output_root / name)
        run_config["feature_cache_dir"] = str(cache_root / name)
        report_path = Path(run_config["output_dir"]) / "oof_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            print(f"reusing_existing_oof={report_path.relative_to(project_root)}")
        else:
            report = run_oof(run_config, config_path)
        if report.get("validation_used") is not False or report.get("composition_file_used") is not False:
            raise RuntimeError(f"candidate {name} did not satisfy train-only isolation")
        reports.append(
            {
                "name": name,
                "run_config": {
                    key: run_config[key]
                    for key in (
                        "static_background_tau_s",
                        "static_motion_gamma",
                        "include_fidelity_contrast",
                        "include_normalized_rho_similarity",
                        "feature_cache_dir",
                    )
                    if key in run_config
                },
                "report": report,
                "ranking_key": list(_ranking_key(report)),
            }
        )
        print(f"candidate_complete={name} mean_event_f1={report['best_oof']['ranking_score']['f1_mean']:.4f}")

    reports.sort(key=lambda row: tuple(row["ranking_key"]), reverse=True)
    winner = reports[0]
    final_config = _final_config(
        base,
        winner["name"],
        winner["run_config"],
        winner["report"],
        project_root,
    )
    summary = {
        "selection_scope": "5-fold GroupKFold OOF on the 13 local train streams",
        "validation_used": False,
        "composition_file_used": False,
        "metric_status": "local package evaluator; train-only candidate selection, not an independent validation or official test score",
        "selected_candidate": winner["name"],
        "selected_config": final_config,
        "candidate_results": [
            {
                "name": row["name"],
                "run_config": row["run_config"],
                "ranking_score": row["report"]["best_oof"]["ranking_score"],
                "by_theta": row["report"]["best_oof"]["by_theta"],
                "selected_decoder": {
                    key: row["report"]["best_oof"][key]
                    for key in (
                        "bias",
                        "transition_cost",
                        "entry_cost",
                        "exit_cost",
                        "min_duration_s",
                        "start_offset_s",
                        "end_offset_s",
                        "duration_profile",
                    )
                },
                "artifact_dir": str((output_root / row["name"]).relative_to(project_root)),
            }
            for row in reports
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    final_config_path.parent.mkdir(parents=True, exist_ok=True)
    final_config_path.write_text(yaml.safe_dump(final_config, sort_keys=False), encoding="utf-8")
    print(f"selected_candidate={winner['name']}")
    print(f"final_config={final_config_path.relative_to(project_root)}")
    print(f"summary={summary_path.relative_to(project_root)}")
    print("validation_used=false composition_file_used=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
