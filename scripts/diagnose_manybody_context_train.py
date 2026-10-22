#!/usr/bin/env python3
"""Áp model R32 đã khóa lên original TRAIN, không refit/chọn ngưỡng.

Đây là in-sample context diagnostic, không phải OOF/test/generalization score.
Không đọc validation hoặc composition. Có thể nối sau job train bằng wait-lock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

import run_manybody_context_nested as context

base = context.base
ROOT = context.ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/manybody_source_context_nested_r32.yaml")
    parser.add_argument("--wait-lock", action="store_true")
    parser.add_argument("--wait-seconds", type=float, default=3600.0)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = ROOT / config["output_dir"]
    deadline = time.monotonic() + args.wait_seconds
    while not (output / "pipeline_lock.json").is_file():
        if not args.wait_lock or time.monotonic() >= deadline:
            raise RuntimeError("model/config must be locked before train-context diagnostic")
        time.sleep(15.0)
    base.signature = context.context_signature
    lock = base.load_lock(config, output)
    destination = output / "original_train_in_sample_diagnostic.json"
    if destination.exists():
        raise RuntimeError("train-context diagnostic already recorded; avoid duplicate scoring")
    with (output / "final_model.pkl").open("rb") as handle:
        model = pickle.load(handle)
    from abc_uwb_falltrack.manybody_readout import predict_readout
    paths = {f"train_{int(path.stem.split('_')[-1]):02d}": path
             for path in sorted((ROOT / config["data_root"] / "streams/train").glob("stream_*.h5"))}
    if len(paths) != 13:
        raise ValueError("expected 13 original train streams")
    # Copy reusable cache; extract() independently validates code/config/source
    # signatures and recalculates if they differ. Never mutate R31 artifacts.
    old_cache = ROOT / "ABC testing/r31/features/original_train_diagnostic"
    new_cache = output / "features/original_train_diagnostic"
    new_cache.mkdir(parents=True, exist_ok=True)
    for key in paths:
        source = old_cache / f"{key}.npz"
        target = new_cache / source.name
        if source.exists() and not target.exists():
            shutil.copy2(source, target)
    with threadpool_limits(limits=int(config.get("compute_threads", 2))):
        packs, names = base.extract(paths, output, lock["feature_config"], "original_train_diagnostic")
        if names != lock["feature_names"]:
            raise ValueError("original train feature schema differs from locked model")
        heads = {key: predict_readout(model, pack["features"]) for key, pack in packs.items()}
        predictions = base.decode_all(heads, packs, names, lock["decoder"], {key: {} for key in packs})
        csv_path = output / "original_train_in_sample_diagnostic.csv"
        base.write_csv(csv_path, predictions)
        np.savez_compressed(output / "original_train_in_sample_heads.npz", **heads)
        # Original train labels only reach scorer after prediction CSV is fixed.
        evaluate = base._package_evaluator(ROOT / config["data_root"])
        metrics = evaluate(base.records(base.events_for(ROOT / config["data_root"], list(paths))), predictions,
                           {key: pack["duration_s"] for key, pack in packs.items()}, theta=0.3)
    result = {"metrics": metrics, "diagnostic_only": True, "not_oof": True,
              "fit_shares_raw_train_source_samples": True, "no_refit": True,
              "no_threshold_change": True, "validation_used": False, "composition_used": False,
              "pipeline_lock_sha256": base.digest(output / "pipeline_lock.json"),
              "prediction_sha256": base.digest(csv_path),
              "diagnostic_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    base.dump(destination, result)
    base.progress(output, "original-train-context-diagnostic", mean_f1=metrics["ranking_score"]["f1_mean"])
    print(json.dumps(metrics["ranking_score"], indent=2), flush=True)


if __name__ == "__main__":
    main()
