#!/usr/bin/env python3
"""Train-only, predeclared readout ablations using main nested-CV selections.

Hai ablation chỉ đổi các cột readout. Cả hai giữ QFI anchor từ full features
trong decoder, do đó không phải đối chứng loại bỏ toàn bộ quantum pipeline.
Không chọn siêu tham số theo điểm ablation, không đọc validation/test hoặc
composition. Script chờ các outer selections mà main runner đã khóa.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from threadpoolctl import threadpool_limits
import yaml

from abc_uwb_falltrack.manybody_decoder import decode_semimarkov
from abc_uwb_falltrack.manybody_readout import fit_readout, predict_readout
from abc_uwb_falltrack.oof import (
    _package_evaluator,
    assert_train_heldout_partition,
    make_group_folds,
)
from run_manybody_nested import boundary, digest, dump, events_for, records, signature, write_csv


ABLATIONS = (
    ("signal_readout_with_qfi_decoder", "signal"),
    ("quantum_readout_with_qfi_decoder", "quantum"),
)


def emit(output: Path, stage: str, **fields) -> None:
    record = {"stage": stage, "utc_epoch": time.time(), **fields}
    dump(output / "progress.json", record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def code_hashes() -> dict[str, str]:
    names = [
        "src/abc_uwb_falltrack/manybody_quantum.py",
        "src/abc_uwb_falltrack/manybody_readout.py",
        "src/abc_uwb_falltrack/manybody_decoder.py",
        "scripts/run_manybody_nested.py",
        "scripts/run_manybody_ablations.py",
    ]
    return {name: digest(ROOT / name) for name in names}


def load_cache(main_output: Path, source_root: Path, feature_config: dict) -> tuple[dict, list, dict]:
    """Chỉ dùng cache source_train; không mở stream HDF5 hoặc mixed val CSV."""
    manifest = json.loads((source_root / "source_group_manifest.json").read_text())
    hashes = {row["stream_id"]: row["source_group_hash"] for row in manifest}
    if len(hashes) != len(manifest) or len(set(hashes.values())) != len(hashes):
        raise ValueError("source groups must have unique hashes")
    feature_hash = digest(ROOT / "src/abc_uwb_falltrack/manybody_quantum.py")
    packs, names = {}, None
    for key in sorted(hashes):
        if not key.startswith("train_"):
            raise ValueError("source manifest must contain train IDs only")
        path = main_output / "features" / "source_train" / f"{key}.npz"
        with np.load(path, allow_pickle=False) as cached:
            metadata = json.loads(str(cached["metadata"].item()))
            if metadata["config"] != feature_config or metadata["code_sha256"] != feature_hash:
                raise ValueError(f"feature cache signature mismatch for {key}")
            these_names = cached["names"].tolist()
            if names is None:
                names = these_names
            x, centers = cached["features"], cached["centers_s"]
            if names != these_names or x.shape != (centers.size, len(names)) or not np.isfinite(x).all():
                raise ValueError(f"feature cache schema mismatch for {key}")
            packs[key] = {"features": x, "centers_s": centers, "duration_s": float(cached["duration_s"])}
    return packs, names, hashes


def subset_packs(packs: dict, ids: list[str], indices: np.ndarray) -> dict:
    return {
        key: {
            "features": np.ascontiguousarray(packs[key]["features"][:, indices]),
            "centers_s": packs[key]["centers_s"],
            "duration_s": packs[key]["duration_s"],
        }
        for key in ids
    }


def aggregate(output: Path, main_output: Path, source_root: Path, packs: dict, evaluate, plan: dict) -> dict:
    """Pool các dự đoán đã khóa; không chọn hoặc fit lại từ kết quả."""
    summary = {
        "experiment": "r31_predeclared_train_only_readout_ablations",
        "plan_sha256": digest(output / "plan.json"),
        "validation_used": False,
        "test_used": False,
        "composition_used": False,
        "hyperparameter_selection": "verbatim main inner-CV selections; no ablation-score selection",
        "shared_qfi_anchor": True,
        "interpretation": "readout feature contribution conditional on fixed full-system hyperparameters and QFI decoder",
        "limitations": [
            "Both ablations retain quantum QFI decoder information; neither is a wholly classical pipeline",
            "Hyperparameters selected for the full representation are not optimized separately for either ablation",
            "Source-path disjointness does not establish subject disjointness",
            "Package-local evaluator on repacked train pseudo-streams; not fresh test evidence or quantum advantage",
        ],
        "ablations": {},
    }
    for name, _ in ABLATIONS:
        fold_reports = []
        all_rows, evaluated_ids = [], []
        for fold in range(plan["outer_splits"]):
            path = output / name / f"fold_{fold}_report.json"
            if not path.exists():
                continue
            fold_report = json.loads(path.read_text())
            prediction_path = output / name / f"fold_{fold}_predictions.csv"
            if digest(prediction_path) != fold_report["prediction_sha256"]:
                raise ValueError("ablation predictions changed after evaluation")
            with prediction_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    all_rows.append({"stream_id": row["stream_id"], "t_start": float(row["t_start"]),
                                     "t_end": float(row["t_end"]), "score": float(row["score"]), "t_alarm": None})
            evaluated_ids.extend(fold_report["heldout_ids"])
            fold_reports.append({"fold": fold, "ranking_score": fold_report["metrics"]["ranking_score"],
                                 "primary": fold_report["metrics"]["primary"]})
        if not fold_reports:
            continue
        if len(evaluated_ids) != len(set(evaluated_ids)):
            raise ValueError("ablation OOF held-out IDs overlap")
        # Every constituent CSV was locked before labels for its fold reached
        # an evaluator. Re-pooling uses exactly those already locked outputs.
        prediction_path = output / name / "oof_predictions.csv"
        write_csv(prediction_path, all_rows)
        prediction_hash = digest(prediction_path)
        dump(output / name / "oof_prediction_lock.json", {
            "prediction_sha256": prediction_hash, "heldout_ids": sorted(evaluated_ids),
            "folds_completed": len(fold_reports), "no_score_based_selection": True,
        })
        metrics = evaluate(records(events_for(source_root, evaluated_ids)), all_rows,
                           {key: packs[key]["duration_s"] for key in evaluated_ids}, theta=0.3)
        summary["ablations"][name] = {
            "feature_count": plan["ablations"][name]["feature_count"],
            "folds_completed": len(fold_reports), "groups_evaluated": len(evaluated_ids),
            "prediction_sha256": prediction_hash, "overall": metrics, "fold_reports": fold_reports,
        }
    dump(output / "ablation_report.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/manybody_source_nested_r31.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    args = parser.parse_args()
    if not 0 < args.poll_seconds <= 30 or args.timeout_seconds <= 0:
        raise ValueError("poll interval must be in (0,30] and timeout positive")
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text())
    main_output = ROOT / config["output_dir"]
    output = ROOT / args.output_dir if args.output_dir else main_output / "ablations"
    source_root = ROOT / config["source_group_root"]
    packs, names, hashes = load_cache(main_output, source_root, config["features"])
    masks = {
        "signal": np.array([i for i, name in enumerate(names) if name.endswith("_radars")]),
        "quantum": np.array([i for i, name in enumerate(names) if not name.endswith("_radars")]),
    }
    if len(names) != 220 or masks["signal"].size != 52 or masks["quantum"].size != 168:
        raise ValueError("predeclared ablation expects 52 signal and 168 quantum feature columns")
    plan_path = output / "plan.json"
    old_plan = json.loads(plan_path.read_text()) if plan_path.exists() else None
    plan = {
        "started_utc_epoch": old_plan["started_utc_epoch"] if old_plan else time.time(),
        "predeclared_before_ablation_metrics": True,
        "config_sha256": digest(config_path), "source_manifest_sha256": digest(source_root / "source_group_manifest.json"),
        "main_run_signature": signature(config), "main_config_snapshot": config,
        "code_sha256": code_hashes(), "outer_splits": int(config["cross_validation"]["outer_splits"]),
        "seed": int(config["seed"]), "validation_used": False, "test_used": False,
        "composition_used": False, "selection_policy": "none; retain both ablations irrespective of scores",
        "shared_settings": "readout, decoder, duration prior and fit seed copied from each main outer inner-CV selection",
        "shared_qfi_anchor": "original full-feature QFI via exactly the main decoder boundary transformation",
        "main_progress_signature_policy": (
            "require embedded pipeline_signature if present; otherwise require atomic outer_progress mtime >= plan start, "
            "plus frozen main signature/code/config and deterministic fold equality; prior progress was archived before this plan"
        ),
        "ablations": {
            name: {"feature_count": int(masks[kind].size), "feature_indices": masks[kind].tolist(),
                   "feature_names": [names[i] for i in masks[kind]],
                   "mask_rule": "endswith('_radars')" if kind == "signal" else "not endswith('_radars')"}
            for name, kind in ABLATIONS
        },
    }
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("ablation plan changed; refuse overwriting predeclaration")
    else:
        dump(plan_path, plan)
    ids = sorted(packs)
    expected_folds = make_group_folds(ids, plan["outer_splits"], plan["seed"])
    evaluate = _package_evaluator(source_root)
    completed = set()
    start = time.monotonic()
    while len(completed) < plan["outer_splits"] * len(ABLATIONS):
        if code_hashes() != plan["code_sha256"] or digest(config_path) != plan["config_sha256"]:
            raise ValueError("code or config changed after ablation predeclaration")
        progress_path = main_output / "outer_progress.json"
        if progress_path.exists():
            main_progress = json.loads(progress_path.read_text())
            embedded_signature = main_progress.get("pipeline_signature")
            if embedded_signature is not None:
                if embedded_signature != plan["main_run_signature"]:
                    raise ValueError("stale main outer_progress signature; refuse ablation fits")
            elif progress_path.stat().st_mtime < plan["started_utc_epoch"]:
                raise ValueError("main outer_progress predates ablation plan; refuse stale folds")
            selections = main_progress["selections"]
        else:
            selections = []
        for selection in selections:
            fold = int(selection["outer"])
            fit_ids, heldout = selection["fit_ids"], selection["heldout_ids"]
            if heldout != expected_folds[fold]:
                raise ValueError("main held-out fold differs from predeclared partition")
            assert_train_heldout_partition(ids, fit_ids, heldout, scope=f"ablation outer {fold}")
            if set(hashes[key] for key in fit_ids) & set(hashes[key] for key in heldout):
                raise ValueError("ablation source overlap")
            readout_config = {**config["readout"], **config["readout_candidates"][selection["readout_index"]]["overrides"]}
            decoder_config = {**config["decoder"], **config["decoder_candidates"][selection["decoder_index"]],
                              **selection["duration_prior"]}
            seed = plan["seed"] + 500 + fold
            selection_hash = hashlib.sha256(json.dumps({"fit_ids": fit_ids, "heldout_ids": heldout,
                "readout": readout_config, "decoder": decoder_config, "seed": seed}, sort_keys=True).encode()).hexdigest()
            for name, kind in ABLATIONS:
                key = (fold, name)
                if key in completed:
                    continue
                folder = output / name
                report_path = folder / f"fold_{fold}_report.json"
                if report_path.exists():
                    old = json.loads(report_path.read_text())
                    if old["selection_sha256"] != selection_hash or digest(folder / f"fold_{fold}_predictions.csv") != old["prediction_sha256"]:
                        raise ValueError("stored ablation fold changed")
                    completed.add(key)
                    continue
                emit(output, "fit", fold=fold, ablation=name)
                fit_events = events_for(source_root, fit_ids)
                sliced = subset_packs(packs, fit_ids, masks[kind])
                model = fit_readout(sliced, fit_events, fit_ids, readout_config, seed)
                del sliced, fit_events
                predictions, heads = [], {}
                for stream in heldout:
                    pack = packs[stream]
                    heads[stream] = predict_readout(model, np.ascontiguousarray(pack["features"][:, masks[kind]]))
                    for event in decode_semimarkov(heads[stream], pack["centers_s"], pack["duration_s"],
                                                  boundary(pack, names), decoder_config):
                        predictions.append({"stream_id": stream, **event, "t_alarm": None})
                prediction_path = folder / f"fold_{fold}_predictions.csv"
                write_csv(prediction_path, predictions)
                prediction_hash = digest(prediction_path)
                model_bytes = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
                model_path = folder / f"fold_{fold}_model.pkl"
                model_path.write_bytes(model_bytes)
                np.savez_compressed(folder / f"fold_{fold}_heads.npz", **heads)
                lock = {
                    "fold": fold, "ablation": name, "selection_sha256": selection_hash,
                    "prediction_sha256": prediction_hash, "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
                    "readout": readout_config, "decoder": decoder_config, "fit_ids": fit_ids, "heldout_ids": heldout,
                    "seed": seed, "source_overlap": 0, "shared_original_qfi_anchor": True,
                    "fit_metadata": model.training_metadata, "prediction_locked_utc_epoch": time.time(),
                    "validation_used": False, "test_used": False, "composition_used": False,
                }
                dump(folder / f"fold_{fold}_prediction_lock.json", lock)
                # First read of this fold's held-out labels for this ablation.
                metrics = evaluate(records(events_for(source_root, heldout)), predictions,
                                   {stream: packs[stream]["duration_s"] for stream in heldout}, theta=0.3)
                dump(report_path, {**lock, "metrics": metrics})
                completed.add(key)
                emit(output, "fold-complete", fold=fold, ablation=name,
                     mean_f1=metrics["ranking_score"]["f1_mean"], completed=len(completed), total=plan["outer_splits"] * len(ABLATIONS))
                del model, model_bytes, heads
                aggregate(output, main_output, source_root, packs, evaluate, plan)
        if len(completed) == plan["outer_splits"] * len(ABLATIONS):
            break
        if time.monotonic() - start > args.timeout_seconds:
            emit(output, "timeout-awaiting-main-folds", completed=len(completed))
            raise TimeoutError("main outer folds did not complete within the ablation wait budget")
        emit(output, "awaiting-main-folds", completed=len(completed), main_folds_available=len(selections))
        time.sleep(args.poll_seconds)
    summary = aggregate(output, main_output, source_root, packs, evaluate, plan)
    emit(output, "complete", results={name: item["overall"]["ranking_score"] for name, item in summary["ablations"].items()})


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
