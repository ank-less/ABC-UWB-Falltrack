#!/usr/bin/env python3
"""R33: predeclare a causal multi-scale readout, reuse frozen R32 caches.

Only R32 train feature/head caches are copied after its pipeline lock passes
the existing context signature. No R32 model, reports or outer predictions
are copied. All model selection remains inside the existing nested runner.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import yaml

import run_manybody_context_nested as context

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CONFIG = ROOT / "configs/manybody_source_context_nested_r32.yaml"
DEFAULT_CONFIG = ROOT / "configs/manybody_source_context_multiscale_r33.yaml"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def progress(output: Path, stage: str, **fields) -> None:
    record = {"stage": stage, "utc_epoch": time.time(), **fields}
    dump(output / "multiscale_launcher_progress.json", record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def frozen_hashes(source_config: Path, target_config: Path) -> dict[str, str]:
    files = [source_config, target_config, Path(__file__),
             ROOT / "scripts/run_manybody_context_nested.py",
             ROOT / "scripts/run_manybody_nested.py",
             ROOT / "scripts/build_source_disjoint_context_package.py",
             ROOT / "src/abc_uwb_falltrack/manybody_quantum.py",
             ROOT / "src/abc_uwb_falltrack/manybody_readout.py",
             ROOT / "src/abc_uwb_falltrack/manybody_decoder.py"]
    return {path.resolve().relative_to(ROOT).as_posix(): digest(path) for path in files}


def verify_config_extension(source: dict, target: dict) -> None:
    expected = copy.deepcopy(source)
    expected["experiment_name"] = "manybody_source_context_multiscale_r33"
    expected["output_dir"] = "ABC testing/r33"
    expected["readout_candidates"].append({
        "name": "compact15_multiscale",
        "overrides": {"max_leaf_nodes": 15, "l2_regularization": 20.0,
                      "temporal_windows_s": [0.4, 1.2]},
    })
    if target != expected:
        raise ValueError("R33 may only extend R32 with the predeclared temporal candidate and its output/name")
    if len(source["decoder_candidates"]) != 18 or source["cross_validation"] != {"outer_splits": 5, "inner_splits": 3}:
        raise ValueError("unexpected R32 grid or nested protocol")


def verify_source_lock(config: dict, output: Path) -> dict:
    lock_path = output / "pipeline_lock.json"
    lock = json.loads(lock_path.read_text())
    if lock["pipeline_signature"] != context.context_signature(config):
        raise ValueError("R32 code/config differs from its pipeline lock")
    if digest(output / "final_model.pkl") != lock["model_sha256"]:
        raise ValueError("R32 model digest differs from its lock")
    manifest = ROOT / config["source_group_root"] / "source_group_manifest.json"
    if digest(manifest) != lock["source_manifest_sha256"]:
        raise ValueError("R32 context manifest differs from its lock")
    if lock["feature_config"] != config["features"]:
        raise ValueError("R32 locked feature configuration differs")
    if lock.get("validation_labels_used", False) or lock.get("test_used", False):
        raise ValueError("R32 source lock must be train-only")
    return lock


def copy_train_caches(source: Path, target: Path, config: dict) -> list[dict]:
    """Independent copies: no symlinks/hardlinks; verify each source/dest hash."""
    locations = [
        (Path("features/source_train"), int(config["context_bundles"])),
        (Path("inner_head_cache"), int(config["cross_validation"]["outer_splits"])
         * int(config["cross_validation"]["inner_splits"]) * len(config["readout_candidates"])),
    ]
    copies = []
    for relative, expected_count in locations:
        source_dir, target_dir = source / relative, target / relative
        files = sorted(source_dir.glob("*.npz"))
        if source_dir.is_symlink() or len(files) != expected_count:
            raise ValueError(f"R32 cache set incomplete or linked: {relative}")
        target_dir.mkdir(parents=True, exist_ok=True)
        for source_file in files:
            if source_file.is_symlink() or not source_file.is_file():
                raise ValueError("R32 cache must be a regular independent file")
            source_digest = digest(source_file)
            destination = target_dir / source_file.name
            if destination.exists():
                if destination.is_symlink() or digest(destination) != source_digest:
                    raise ValueError("refusing to overwrite an existing different R33 cache")
            else:
                shutil.copy2(source_file, destination)
            if source_file.samefile(destination) or digest(source_file) != source_digest or digest(destination) != source_digest:
                raise ValueError("cache copy changed or shares source identity")
            copies.append({"path": (relative / source_file.name).as_posix(), "sha256": source_digest,
                           "bytes": source_file.stat().st_size, "independent_copy": True})
    return copies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    args = parser.parse_args()
    if not 0 < args.poll_seconds <= 30 or args.timeout_seconds <= 0:
        raise ValueError("poll seconds must be in (0,30] and timeout positive")
    source_config, target_config = args.source_config.resolve(), args.config.resolve()
    source = yaml.safe_load(source_config.read_text())
    target = yaml.safe_load(target_config.read_text())
    verify_config_extension(source, target)
    source_output, target_output = ROOT / source["output_dir"], ROOT / target["output_dir"]
    plan_path = target_output / "predeclared_plan.json"
    old_plan = json.loads(plan_path.read_text()) if plan_path.exists() else None
    if old_plan is None and (source_output / "nested_oof_report.json").exists():
        raise ValueError("R32 pooled report already exists; cannot claim advance predeclaration")
    plan = {
        "experiment": target["experiment_name"],
        "started_utc_epoch": old_plan["started_utc_epoch"] if old_plan else time.time(),
        "predeclared_before_r32_pooled_results": True,
        "source_config_snapshot": source, "target_config_snapshot": target,
        "frozen_sha256": frozen_hashes(source_config, target_config),
        "r32_context_signature": context.context_signature(source),
        "r33_context_signature": context.context_signature(target),
        "only_new_readout": "compact15_multiscale: same15 leaves/L2=20, temporal windows [0.4,1.2] s",
        "rationale": "A fall lasts approximately1.7s; adding a causal1.2s summary may capture structure missed by the0.4s readout",
        "selection_policy": "existing train-only inner CV; no validation/test or outer-score configuration selection",
        "cache_reuse": "after verified R32 pipeline lock, independently copy only features/source_train and inner_head_cache",
        "cache_revalidation": "existing runner checks readout/feature config, fit and heldout IDs, seed, manifest and model code hashes",
        "validation_used": False, "test_used": False, "composition_used": False,
    }
    if old_plan is not None and old_plan != plan:
        raise ValueError("R33 predeclaration changed; refusing to overwrite")
    if old_plan is None:
        dump(plan_path, plan)
    progress(target_output, "predeclared-waiting-r32-lock")
    started_wait = time.monotonic()
    while not (source_output / "pipeline_lock.json").exists():
        if frozen_hashes(source_config, target_config) != plan["frozen_sha256"]:
            raise ValueError("code/config changed during R33 wait")
        if time.monotonic() - started_wait > args.timeout_seconds:
            raise TimeoutError("R32 pipeline lock did not arrive within the wait budget")
        time.sleep(args.poll_seconds)
    if frozen_hashes(source_config, target_config) != plan["frozen_sha256"]:
        raise ValueError("code/config changed before R33 launch")
    lock = verify_source_lock(source, source_output)
    lock_hash = digest(source_output / "pipeline_lock.json")
    progress(target_output, "copying-frozen-r32-train-caches")
    copies = copy_train_caches(source_output, target_output, source)
    if digest(source_output / "pipeline_lock.json") != lock_hash:
        raise ValueError("R32 lock changed during cache copying")
    verify_source_lock(source, source_output)
    dump(target_output / "cache_reuse_manifest.json", {
        "source_pipeline_lock_sha256": lock_hash, "source_pipeline_signature": lock["pipeline_signature"],
        "copied_files": copies, "copied_file_count": len(copies),
        "hardlinks_used": False, "symlinks_used": False,
        "models_locks_reports_outer_outputs_copied": False,
    })
    progress(target_output, "launching-existing-context-runner", copied_files=len(copies))
    # Only CLI arguments are set. Existing context/main functions own their
    # normal signature, resource limits, nested CV, cache validation and lock.
    sys.argv = [str(ROOT / "scripts/run_manybody_context_nested.py"), "--config", str(target_config)]
    context.main()
    verify_source_lock(target, target_output)
    progress(target_output, "r33-pipeline-locked", pipeline_signature=context.context_signature(target))


if __name__ == "__main__":
    main()
