#!/usr/bin/env python3
"""Train-only nested source-group selection and a separately locked val run.

No stream composition is read. The train-only repack and source manifest must
already exist. Validation labels are accessible only in evaluate-validation,
after immutable model/config/prediction digests have been written.
"""
from __future__ import annotations

import argparse
from collections import Counter
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

import h5py
import numpy as np
import yaml
from threadpoolctl import threadpool_limits

from abc_uwb_falltrack.oof import (
    _package_evaluator, assert_fold_partition, assert_train_heldout_partition,
    make_group_folds, read_train_events,
)


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json), encoding="utf-8")
    temp.replace(path)


def _json(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def signature(config):
    h = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for name in ("manybody_quantum.py", "manybody_readout.py", "manybody_decoder.py"):
        h.update((ROOT / "src" / "abc_uwb_falltrack" / name).read_bytes())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()


def progress(output, stage, **extra):
    record = {"stage": stage, "utc_epoch": time.time(), **extra}
    dump(output / "progress.json", record)
    print(json.dumps(record, ensure_ascii=False, default=_json), flush=True)


def events_for(data_root, ids):
    result = {key: [] for key in ids}
    for event in read_train_events(data_root / "streams_events.csv", set(ids)):
        result[event.stream_id].append(event)
    return result


def records(events):
    return [{"stream_id": key, "t_start": e.t_start, "t_end": e.t_end, "ignore": e.ignore}
            for key, values in events.items() for e in values]


def duration_prior(events):
    durations = np.array([e.t_end - e.t_start for v in events.values() for e in v if not e.ignore])
    if not len(durations):
        raise ValueError("no evaluable fit events for duration prior")
    logs = np.log(durations)
    return {"duration_median_s": float(np.median(durations)),
            "duration_log_sigma": float(max(0.25, 1.4826 * np.median(np.abs(logs - np.median(logs)))))}


def check_source_manifest(source_root):
    manifest = json.loads((source_root / "source_group_manifest.json").read_text())
    mapping = {row["stream_id"]: row["source_group_hash"] for row in manifest}
    if len(mapping) != len(manifest) or len(set(mapping.values())) != len(mapping):
        raise ValueError("source manifest must contain one unique source-path hash per pseudo-stream")
    paths = {}
    for path in sorted((source_root / "streams" / "train").glob("stream_*.h5")):
        key = f"train_{int(path.stem.split('_')[-1]):02d}"
        with h5py.File(path, "r") as handle:
            if str(handle.attrs["source_group_hash"]) != mapping[key]:
                raise ValueError("manifest and HDF5 group hash disagree")
        paths[key] = path
    if set(mapping) != set(paths):
        raise ValueError("source manifest does not cover HDF5 files")
    return mapping, paths


def extract(paths, output, feature_config, split):
    from abc_uwb_falltrack.manybody_quantum import extract_manybody_features
    cache = output / "features" / split
    cache.mkdir(parents=True, exist_ok=True)
    feature_code_hash = digest(ROOT / "src" / "abc_uwb_falltrack" / "manybody_quantum.py")
    packs, names, diagnostics = {}, None, {}
    for index, (key, path) in enumerate(sorted(paths.items())):
        st = path.stat()
        meta = {"config": feature_config, "code_sha256": feature_code_hash,
                "source_name": path.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns}
        dest = cache / f"{key}.npz"
        loaded = False
        if dest.exists():
            with np.load(dest, allow_pickle=False) as old:
                if json.loads(str(old["metadata"].item())) == meta:
                    x, centers = old["features"], old["centers_s"]
                    duration = float(old["duration_s"])
                    these_names = old["names"].tolist()
                    diag = json.loads(str(old["diagnostics"].item()))
                    loaded = True
        if not loaded:
            x, centers, duration, these_names, diag = extract_manybody_features(path, feature_config)
            np.savez_compressed(dest, features=x, centers_s=centers, duration_s=duration,
                                names=np.array(these_names), metadata=np.array(json.dumps(meta)),
                                diagnostics=np.array(json.dumps(diag, default=_json)))
        if names is None:
            names = these_names
        if names != these_names or x.shape != (len(centers), len(names)) or not np.isfinite(x).all():
            raise ValueError(f"invalid feature schema: {key}")
        packs[key] = {"features": x, "centers_s": centers, "duration_s": duration}
        diagnostics[key] = diag
        if index % 20 == 0 or index + 1 == len(paths):
            progress(output, f"extract-{split}", completed=index + 1, total=len(paths))
    dump(output / f"{split}_feature_diagnostics.json", {"names": names, "streams": diagnostics})
    return packs, names


def boundary(pack, names):
    indices = [i for i, name in enumerate(names) if "qfi" in name.lower()]
    if not indices:
        raise ValueError("QFI feature required for boundary pinning")
    values = np.maximum(pack["features"][:, indices].astype(float), 0.0)
    # Fixed pointwise bounded map; no holdout-wide normalization/statistics.
    values = np.log1p(values).mean(axis=1)
    return values / (1.0 + values)


def decode_all(heads, packs, names, config, priors):
    from abc_uwb_falltrack.manybody_decoder import decode_semimarkov
    predictions = []
    for key in sorted(heads):
        options = {**config, **priors[key]}
        for event in decode_semimarkov(heads[key], packs[key]["centers_s"],
                                      packs[key]["duration_s"], boundary(packs[key], names), options):
            predictions.append({"stream_id": key, **event, "t_alarm": None})
    return predictions


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stream_id", "t_start", "t_end", "score"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rank(report):
    start = report["ranking_score"].get("start_error_median_abs")
    fa = report["ranking_score"].get("false_alarms_per_hour")
    return (float(report["ranking_score"]["f1_mean"]),
            -float(start) if start is not None else -1e9,
            -float(fa) if fa is not None else -1e9)


def fit_scoped(packs, source_root, ids, config, seed):
    from abc_uwb_falltrack.manybody_readout import fit_readout
    fit_events = events_for(source_root, ids)
    model = fit_readout({key: packs[key] for key in ids}, fit_events, ids, config, seed)
    return model, duration_prior(fit_events)


def train(config, output):
    from abc_uwb_falltrack.manybody_readout import predict_readout
    source_root = ROOT / config["source_group_root"]
    hashes, paths = check_source_manifest(source_root)
    packs, names = extract(paths, output, config["features"], "source_train")
    ids = sorted(packs)
    folds = make_group_folds(ids, int(config["cross_validation"]["outer_splits"]), int(config["seed"]))
    assert_fold_partition(ids, folds, scope="manybody outer")
    evaluate = _package_evaluator(source_root)
    all_predictions, selections, outer_reports = [], [], []
    all_heads = {}
    candidates = config["readout_candidates"]
    for outer_index, heldout in enumerate(folds):
        fit_ids = sorted(set(ids) - set(heldout))
        assert_train_heldout_partition(ids, fit_ids, heldout, scope=f"outer {outer_index}")
        if set(hashes[key] for key in fit_ids) & set(hashes[key] for key in heldout):
            raise ValueError("outer source overlap")
        inner_folds = make_group_folds(fit_ids, int(config["cross_validation"]["inner_splits"]),
                                       int(config["seed"]) + 100 + outer_index)
        candidate_rows = []
        best = None
        for candidate_index, candidate in enumerate(candidates):
            readout_config = {**config["readout"], **candidate["overrides"]}
            inner_heads, priors, fit_scopes = {}, {}, []
            for inner_index, inner_heldout in enumerate(inner_folds):
                inner_fit = sorted(set(fit_ids) - set(inner_heldout))
                assert_train_heldout_partition(fit_ids, inner_fit, inner_heldout, scope="inner")
                if set(hashes[key] for key in inner_fit) & set(hashes[key] for key in inner_heldout):
                    raise ValueError("inner source overlap")
                fit_seed = int(config["seed"]) + outer_index * 100 + inner_index
                cache_signature = {"readout": readout_config, "fit_ids": inner_fit,
                                   "heldout_ids": inner_heldout, "seed": fit_seed,
                                   "features": config["features"],
                                   "manifest_sha256": digest(source_root / "source_group_manifest.json"),
                                   "readout_sha256": digest(ROOT / "src/abc_uwb_falltrack/manybody_readout.py"),
                                   "quantum_sha256": digest(ROOT / "src/abc_uwb_falltrack/manybody_quantum.py")}
                head_cache = output / "inner_head_cache" / f"outer_{outer_index}_{candidate['name']}_{inner_index}.npz"
                head_cache.parent.mkdir(parents=True, exist_ok=True)
                cached_heads = None
                if head_cache.exists():
                    with np.load(head_cache, allow_pickle=False) as cached:
                        if json.loads(str(cached["signature"].item())) == cache_signature:
                            cached_heads = {key: cached[key] for key in inner_heldout}
                            prior = json.loads(str(cached["prior"].item()))
                            fit_metadata = json.loads(str(cached["fit_metadata"].item()))
                if cached_heads is None:
                    model, prior = fit_scoped(packs, source_root, inner_fit, readout_config, fit_seed)
                    cached_heads = {key: predict_readout(model, packs[key]["features"]) for key in inner_heldout}
                    fit_metadata = getattr(model, "training_metadata", {})
                    np.savez_compressed(head_cache, **cached_heads,
                                        signature=np.array(json.dumps(cache_signature, default=_json)),
                                        prior=np.array(json.dumps(prior, default=_json)),
                                        fit_metadata=np.array(json.dumps(fit_metadata, default=_json)))
                for key in inner_heldout:
                    if cached_heads[key].shape != (len(packs[key]["centers_s"]), 3) or not np.isfinite(cached_heads[key]).all():
                        raise ValueError("invalid cached inner predictions")
                    inner_heads[key] = cached_heads[key]
                    priors[key] = prior
                fit_scopes.append({"fit_ids": inner_fit, "heldout_ids": inner_heldout,
                                   "duration_prior": prior,
                                   "fit_metadata": fit_metadata,
                                   "prediction_cache_sha256": digest(head_cache)})
                progress(output, "inner-fit", outer=outer_index, candidate=candidate["name"], inner=inner_index)
            # Inner-heldout labels are first used here, after predictions lock.
            gt_inner = records(events_for(source_root, fit_ids))
            durations = {key: packs[key]["duration_s"] for key in fit_ids}
            for decoder_index, decoder in enumerate(config["decoder_candidates"]):
                decoder_config = {**config["decoder"], **decoder}
                preds = decode_all(inner_heads, packs, names, decoder_config, priors)
                metrics = evaluate(gt_inner, preds, durations, theta=0.3)
                row = {"readout_index": candidate_index, "readout_name": candidate["name"],
                       "decoder_index": decoder_index, "ranking_score": metrics["ranking_score"],
                       "primary": metrics["primary"], "by_theta": metrics["by_theta"]}
                candidate_rows.append(row)
                if best is None or rank(metrics) > best[0]:
                    best = (rank(metrics), candidate_index, decoder_index)
            dump(output / "search" / f"outer_{outer_index}_{candidate['name']}.json",
                 {"fit_scopes": fit_scopes, "rows": candidate_rows})
        _, model_index, decoder_index = best
        readout_config = {**config["readout"], **candidates[model_index]["overrides"]}
        model, prior = fit_scoped(packs, source_root, fit_ids, readout_config, int(config["seed"]) + 500 + outer_index)
        heads = {key: predict_readout(model, packs[key]["features"]) for key in heldout}
        predictions = decode_all(heads, packs, names,
                                 {**config["decoder"], **config["decoder_candidates"][decoder_index]},
                                 {key: prior for key in heldout})
        write_csv(output / f"outer_{outer_index}_predictions.csv", predictions)
        all_predictions.extend(predictions)
        all_heads.update(heads)
        selection = {"outer": outer_index, "readout_index": model_index,
                     "readout_name": candidates[model_index]["name"], "decoder_index": decoder_index,
                     "fit_ids": fit_ids, "heldout_ids": heldout, "source_overlap": 0,
                     "duration_prior": prior, "fit_metadata": getattr(model, "training_metadata", {})}
        selections.append(selection)
        # Outer-heldout labels only reach this evaluator after output is written.
        metrics = evaluate(records(events_for(source_root, heldout)), predictions,
                           {key: packs[key]["duration_s"] for key in heldout}, theta=0.3)
        outer_reports.append({"fold": outer_index, "metrics": metrics})
        dump(output / "outer_progress.json", {"selections": selections, "fold_reports": outer_reports})
        progress(output, "outer-complete", outer=outer_index,
                 mean_f1=metrics["ranking_score"]["f1_mean"], selected_readout=model_index, selected_decoder=decoder_index)
    if set(all_heads) != set(ids):
        raise ValueError("OOF coverage incomplete")
    write_csv(output / "nested_oof_predictions.csv", all_predictions)
    np.savez_compressed(output / "nested_oof_heads.npz", **all_heads)
    metrics = evaluate(records(events_for(source_root, ids)), all_predictions,
                       {key: packs[key]["duration_s"] for key in ids}, theta=0.3)
    report = {"experiment": config["experiment_name"], "overall": metrics,
              "selections": selections, "fold_reports": outer_reports,
              "source_group_count": len(ids), "validation_used": False, "test_used": False,
              "composition_used_for_fit_or_inference": False,
              "limitations": ["Source-path-disjoint, not proven subject-disjoint",
                              "Repacked pseudo-stream metric, not original official continuous-stream split",
                              "Package-local evaluator; official ranking protocol unconfirmed",
                              "Model development has reused this train package; no fresh-test claim"]}
    dump(output / "nested_oof_report.json", report)
    # Predeclared mode of INNER selections. Outer scores never select the final config.
    pair = Counter((r["readout_index"], r["decoder_index"]) for r in selections).most_common(1)[0][0]
    chosen_readout = {**config["readout"], **candidates[pair[0]]["overrides"]}
    chosen_decoder = {**config["decoder"], **config["decoder_candidates"][pair[1]]}
    model, prior = fit_scoped(packs, source_root, ids, chosen_readout, int(config["seed"]) + 1000)
    model_path = output / "final_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    locked = {"experiment": config["experiment_name"], "pipeline_signature": signature(config),
              "model_sha256": digest(model_path), "readout": chosen_readout,
              "source_manifest_sha256": digest(source_root / "source_group_manifest.json"),
              "package_evaluator_sha256": digest(source_root / "src/abc2027/evaluate.py"),
              "decoder": {**chosen_decoder, **prior}, "feature_config": config["features"],
              "feature_names": names, "selection": "mode of inner-CV selected pairs; tie resolved by fold order",
              "selection_pair": pair, "source_group_count": len(ids), "fit_duration_prior": prior,
              "validation_labels_used": False, "test_used": False,
              "validation_policy": config["validation_policy"], "locked_utc_epoch": time.time()}
    dump(output / "pipeline_lock.json", locked)
    progress(output, "pipeline-locked", mean_oof_f1=metrics["ranking_score"]["f1_mean"], pair=pair)


def load_lock(config, output):
    lock = json.loads((output / "pipeline_lock.json").read_text())
    if lock["pipeline_signature"] != signature(config) or lock["model_sha256"] != digest(output / "final_model.pkl"):
        raise ValueError("pipeline changed after lock; validation use forbidden")
    return lock


def predict_validation(config, output):
    from abc_uwb_falltrack.manybody_readout import predict_readout
    lock = load_lock(config, output)
    if (output / "validation_prediction_lock.json").exists():
        raise ValueError("validation prediction already locked; do not rerun")
    with (output / "final_model.pkl").open("rb") as handle:
        model = pickle.load(handle)
    paths = {f"val_{int(p.stem.split('_')[-1]):02d}": p
             for p in sorted((ROOT / config["data_root"] / "streams" / "val").glob("stream_*.h5"))}
    if len(paths) != 5:
        raise ValueError("expected five local validation streams")
    packs, names = extract(paths, output, lock["feature_config"], "validation")
    if names != lock["feature_names"]:
        raise ValueError("validation feature schema differs")
    heads = {key: predict_readout(model, pack["features"]) for key, pack in packs.items()}
    predictions = decode_all(heads, packs, names, lock["decoder"], {key: {} for key in packs})
    pred_path = ROOT / config["submission_dir"] / f"{config['experiment_name']}_val.csv"
    write_csv(pred_path, predictions)
    write_csv(output / "validation_predictions.csv", predictions)
    np.savez_compressed(output / "validation_heads.npz", **heads)
    dump(output / "validation_prediction_lock.json",
         {"pipeline_lock_sha256": digest(output / "pipeline_lock.json"), "prediction_sha256": digest(pred_path),
          "prediction_path": str(pred_path.relative_to(ROOT)), "prediction_labels_read": False,
          "composition_used": False, "durations": {k: p["duration_s"] for k, p in packs.items()},
          "locked_utc_epoch": time.time()})
    progress(output, "validation-predictions-locked", predictions=len(predictions))


def evaluate_validation(config, output):
    load_lock(config, output)
    destination = output / "validation_metrics.json"
    if destination.exists():
        raise ValueError("validation already evaluated; no iterative validation selection")
    lock = json.loads((output / "validation_prediction_lock.json").read_text())
    pred_path = ROOT / lock["prediction_path"]
    if lock["prediction_sha256"] != digest(pred_path) or lock["pipeline_lock_sha256"] != digest(output / "pipeline_lock.json"):
        raise ValueError("predictions/pipeline changed before evaluation")
    data_root = ROOT / config["data_root"]
    sys.path.insert(0, str(data_root / "src"))
    from abc2027.evaluate import evaluate, load_ground_truth, validate_predictions
    predictions = validate_predictions(pred_path, lock["durations"])
    metric = evaluate(load_ground_truth(data_root / "streams_events.csv", ignore_flagged=True),
                      predictions, lock["durations"], theta=0.3)
    dump(destination, {"metrics": metric, "diagnostic_only": True,
                       "validation_previously_consumed": True, "pipeline_sha256": lock["pipeline_lock_sha256"],
                       "prediction_sha256": lock["prediction_sha256"], "no_selection_from_this_score": True,
                       "target_achieved": metric["ranking_score"]["f1_mean"] >= float(config["validation_target"])})
    progress(output, "validation-evaluated", mean_f1=metric["ranking_score"]["f1_mean"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/manybody_source_nested_r31.yaml")
    parser.add_argument("--stage", choices=["train", "predict-validation", "evaluate-validation"], default="train")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = ROOT / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "train" and (output / "validation_prediction_lock.json").exists():
        raise ValueError("this experiment has already reached validation; new fitting forbidden")
    with threadpool_limits(limits=2):
        {"train": train, "predict-validation": predict_validation,
         "evaluate-validation": evaluate_validation}[args.stage](config, output)


if __name__ == "__main__":
    main()
