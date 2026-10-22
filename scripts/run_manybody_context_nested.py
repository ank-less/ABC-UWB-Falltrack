#!/usr/bin/env python3
"""R32: source-disjoint continuous contexts, train-only nested selection.

Không chấm lại validation đã tiêu thụ. Wrapper kiểm tra toàn bộ source sets
trong các bundle trước khi dùng runner nested đã có. Code/config/model lock
bao gồm cả wrapper và builder để giữ provenance của protocol mới.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

import run_manybody_nested as base

ROOT = Path(__file__).resolve().parents[1]
_BASE_SIGNATURE = base.signature


def context_signature(config):
    digest = hashlib.sha256(_BASE_SIGNATURE(config).encode())
    digest.update(Path(__file__).read_bytes())
    digest.update((ROOT / "scripts/build_source_disjoint_context_package.py").read_bytes())
    return digest.hexdigest()


def validate_context_provenance(config):
    package = ROOT / config["source_group_root"]
    manifest = json.loads((package / "source_group_manifest.json").read_text())
    expected_manifest = json.loads((ROOT / config["original_source_group_root"] / "source_group_manifest.json").read_text())
    expected_sources = {row["source_group_hash"] for row in expected_manifest}
    seen = set()
    source_sets = {}
    for row in manifest:
        sources = row.get("source_hashes", [])
        source_set = set(sources)
        if not sources or len(source_set) != len(sources) or seen & source_set:
            raise ValueError("context bundles must have nonempty, mutually disjoint source-path sets")
        seen.update(source_set)
        source_sets[row["stream_id"]] = sorted(source_set)
    if seen != expected_sources:
        raise ValueError("context bundles do not cover exactly the original train-only source groups")
    if len(manifest) != int(config["context_bundles"]):
        raise ValueError("unexpected context bundle count")
    mapping, paths = base.check_source_manifest(package)
    ids = sorted(mapping)
    outer_folds = base.make_group_folds(ids, int(config["cross_validation"]["outer_splits"]), int(config["seed"]))
    checks = []
    for fold_index, heldout in enumerate(outer_folds):
        fit_ids = sorted(set(ids) - set(heldout))
        fit_sources = set().union(*(set(source_sets[key]) for key in fit_ids))
        heldout_sources = set().union(*(set(source_sets[key]) for key in heldout))
        if fit_sources & heldout_sources:
            raise ValueError("underlying source file appears across outer folds")
        inner_folds = base.make_group_folds(fit_ids, int(config["cross_validation"]["inner_splits"]),
                                            int(config["seed"]) + 100 + fold_index)
        for inner_heldout in inner_folds:
            inner_fit = sorted(set(fit_ids) - set(inner_heldout))
            inner_fit_sources = set().union(*(set(source_sets[key]) for key in inner_fit))
            inner_heldout_sources = set().union(*(set(source_sets[key]) for key in inner_heldout))
            if inner_fit_sources & inner_heldout_sources:
                raise ValueError("underlying source file appears across inner folds")
        checks.append({"outer": fold_index, "fit_source_count": len(fit_sources),
                       "heldout_source_count": len(heldout_sources), "source_overlap": 0})
    result = {"passed": True, "context_bundles": len(manifest), "original_source_path_groups": len(seen),
              "coverage_exact": True, "outer_and_inner_source_overlap": 0,
              "folds": checks, "validation_used": False, "test_used": False,
              "limitations": ["Source-path-disjoint, not verified participant-disjoint",
                              "Train-only synthetic continuous contexts, not official original streams"]}
    base.dump(ROOT / config["output_dir"] / "context_provenance_integrity.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/manybody_source_context_nested_r32.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    validate_context_provenance(config)
    base.signature = context_signature
    configured_threads = int(config.get("compute_threads", 2))
    if configured_threads < 1:
        raise ValueError("compute_threads must be positive")
    original_limits = base.threadpool_limits

    def runtime_limits(*, limits):
        return original_limits(limits=configured_threads)

    base.threadpool_limits = runtime_limits
    sys.argv = [str(__file__), "--config", str(args.config), "--stage", "train"]
    base.main()


if __name__ == "__main__":
    main()
