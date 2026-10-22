"""Small synthetic package with the same layout as the real one, for tests and CLI checks.

The numbers are made up. Nothing here describes the real data.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

TRAIN_FALL_PEOPLE = (1, 2, 3)
TRAIN_ADL_SUBJECTS = (4, 5, 6)
VAL_FALL_PEOPLE = (8,)
VAL_ADL_SUBJECTS = (11,)


def make_synthetic_package(
    root: Path,
    n_train: int = 3,
    n_val: int = 2,
    seconds: float = 120.0,
    rate_hz: float = 50.0,
    n_bins: int = 16,
    seed: int = 0,
) -> Path:
    """Write ``streams/{train,val}/stream_XX.h5``, ``streams_events.csv`` and
    ``streams_composition.csv`` under ``root``; return ``root``."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    events, composition = [], []
    for split, count, people, subjects in (
        ("train", n_train, TRAIN_FALL_PEOPLE, TRAIN_ADL_SUBJECTS),
        ("val", n_val, VAL_FALL_PEOPLE, VAL_ADL_SUBJECTS),
    ):
        folder = root / "streams" / split
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            stream_id = f"{split}_{index:02d}"
            frames = int(seconds * rate_hz)
            t = np.arange(frames) / rate_hz
            base = rng.normal(0.0, 1.0, size=(3, 1, n_bins)) * 1e-3
            cir = base + rng.normal(0.0, 2e-5, size=(3, frames, n_bins))
            cursor, seg = 0.0, 0
            while cursor < seconds - 1e-9:
                is_fall = seg % 2 == 1
                length = float(rng.uniform(8.0, 14.0)) if is_fall else float(rng.uniform(10.0, 20.0))
                end = min(seconds, cursor + length)
                if is_fall:
                    person = people[(seg // 2) % len(people)]
                    dataset = 1 + seg % 3
                    path = f"falls/Dataset {dataset}/Test_P{dataset}_{person}/Chute0{1 + seg % 4}_{seg}.json"
                    onset = cursor + min(3.0, 0.3 * (end - cursor))
                    fall_end = min(end - 0.5, onset + float(rng.uniform(0.8, 2.5)))
                    if fall_end - onset > 0.3:
                        a, b = int(onset * rate_hz), int(fall_end * rate_hz)
                        cir[:, a:b, :] += rng.normal(0.0, 3e-4, size=(3, b - a, n_bins))
                        events.append(
                            {
                                "stream_id": stream_id,
                                "event_id": f"{stream_id}_{seg:04d}",
                                "t_start": round(onset, 2),
                                "t_end": round(fall_end, 2),
                                "fall_type": f"Chute0{1 + seg % 4}",
                                "ignore": "false",
                                "quality": "low" if seg % 7 == 3 else "normal",
                                "needs_review": "false",
                                "source_path": path,
                                "source_t_start": 0.0,
                                "source_t_end": 0.0,
                                "dataset": f"Dataset {dataset}",
                            }
                        )
                    kind, label = "fall", f"Chute0{1 + seg % 4}"
                else:
                    subject = subjects[(seg // 2) % len(subjects)]
                    label = ("Marcher", "Manger", "Lire_livre")[seg % 3]
                    path = f"activities/{label}_{subject}.json"
                    a, b = int(cursor * rate_hz), int(end * rate_hz)
                    cir[:, a:b, :] += rng.normal(0.0, 4e-5, size=(3, b - a, n_bins))
                    kind = "adl"
                composition.append(
                    {
                        "stream_id": stream_id,
                        "segment_id": f"{stream_id}_{seg:04d}",
                        "kind": kind,
                        "path": path,
                        "label": label,
                        "dataset": "",
                        "t_start": round(cursor, 2),
                        "t_end": round(end, 2),
                        "source_t0": 0.0,
                        "source_t1": round(end - cursor, 2),
                    }
                )
                cursor, seg = end, seg + 1
            with h5py.File(folder / f"stream_{index:02d}.h5", "w") as handle:
                handle.create_dataset("cir", data=cir.astype(np.float32), chunks=(3, min(500, frames), n_bins))
                handle.create_dataset("t", data=t)
                handle.create_dataset("radar_serials", data=np.array([29171, 29444, 30722], dtype=np.int32))
                handle.attrs["split"] = split
                handle.attrs["stream_index"] = index
    pd.DataFrame(events).to_csv(root / "streams_events.csv", index=False)
    pd.DataFrame(composition).to_csv(root / "streams_composition.csv", index=False)
    return root


def synthetic_config(root: Path, output_root: Path, n_bins: int = 16) -> dict:
    """A complete config pointing at a synthetic package (provisional event metrics)."""
    root, output_root = Path(root), Path(output_root)
    return {
        "data": {
            "root": root.as_posix(),
            "streams_dir": "streams",
            "events_csv": "streams_events.csv",
            "composition_csv": "streams_composition.csv",
            "manifest_path": (output_root / "manifest" / "manifest.csv").as_posix(),
            "segments_path": (output_root / "manifest" / "segments.csv").as_posix(),
            "include_splits": ["train", "val"],
            "rate_hz": 50.0,
            "expected_radars": 3,
            "expected_bins": n_bins,
            "chunk_frames": 700,
            "official_split_field": "official_split",
            "recording_field": "stream_id",
            "group_field": "subject_group",
        },
        "validation": {"strict": True, "check_timestamps": True, "check_missing": True, "check_split_overlap": True},
        "split": {"strategy": "official", "holdout_split": "val", "group_key": "subject_group", "n_folds": 3, "seed": 7},
        "preprocessing": {"background_tau_seconds": 3.0},
        "windowing": {
            "duration_seconds": 1.0,
            "stride_seconds": 0.2,
            "minimum_valid_fraction": 1.0,
            "drop_last": True,
            "positive_overlap_threshold": 0.5,
            "negative_overlap_max": 0.0,
            "drop_ambiguous": True,
            "flagged_event_policy": "positive",
        },
        "features": {
            "representation": "motion_energy",
            "series": ["bg_energy", "diff_energy"],
            "log_transform": True,
            "enabled_features": ["mean", "std", "max", "median", "slope"],
            "cross_radar": True,
            "context_seconds": 5.0,
        },
        "model": {
            "candidates": [{"name": "dummy"}, {"name": "logistic", "parameters": {"C": [1.0]}}],
            "class_weight": "balanced",
        },
        "threshold": {"selection_metric": "official_f1_mean", "candidate_values": [0.5, 0.8]},
        "postprocessing": {
            "smoothing_windows": [1],
            "minimum_positive_windows": [1],
            "merge_gap_seconds": [0.4],
            "minimum_event_duration_seconds": [0.4],
            "cooldown_seconds": [0.0],
            "interval_mode": "center",
        },
        "evaluation": {"official_evaluator": None, "ignore_flagged": True, "report_by_group": True},
        "diagnostics": {"learning_curve_fractions": [0.5, 1.0], "junction_margin_seconds": 0.5, "suspicious_score": 0.95},
        "experiment": {
            "name": "synthetic",
            "seed": 7,
            "output_root": (output_root / "experiments").as_posix(),
            "cache_dir": None,
            "max_streams_per_split": None,
            "max_seconds_per_stream": None,
        },
    }
