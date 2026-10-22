"""Generate the EDA/preprocessing figures and numbers (NV2 report) from the dataset package.

Writes PNG figures to reports/figures/ and the numbers to reports/eda_stats.json.
Signal and feature analysis uses the official train split only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap
from abc_uwb_falltrack import eda
from abc_uwb_falltrack.config import get, load_config
from abc_uwb_falltrack.experiment import setup_logging, write_json
from abc_uwb_falltrack.manifest import build_manifest
from abc_uwb_falltrack.preprocessing import cached_frame_series


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _bootstrap.add_common_arguments(parser)
    parser.add_argument("--figures", default="reports/figures", help="figure folder (default: %(default)s)")
    parser.add_argument("--stats", default="reports/eda_stats.json", help="numbers JSON (default: %(default)s)")
    parser.add_argument("--inspection", default=None, help="inspection.json from inspect_data.py, to copy value statistics")
    parser.add_argument("--sample-streams", default="train_00,train_01,train_02", help="train streams for the slower analyses")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.override, args.smoke)
    setup_logging()
    figures = Path(args.figures)
    sample = args.sample_streams.split(",")
    rate = float(get(cfg, "data.rate_hz"))

    manifest, segments, events = build_manifest(cfg)
    stats: dict = {"note": "descriptive counts: train + val; signal/feature analysis: train only"}
    stats["overview"] = eda.overview_table(manifest, events, segments).to_dict(orient="records")
    lengths = events["t_end"] - events["t_start"]
    stats["fall_duration_s"] = {k: round(float(v), 3) for k, v in lengths.describe(percentiles=[0.1, 0.5, 0.9]).items()}
    stats["fall_type"] = events.groupby([eda.split_of(events["stream_id"]), "fall_type"]).size().unstack(fill_value=0).to_dict()
    stats["dataset"] = events.groupby([eda.split_of(events["stream_id"]), "dataset"]).size().unstack(fill_value=0).to_dict()
    stats["streams_with_each_adl_file_train"] = (
        segments[(segments["official_split"] == "train") & (segments["kind"] == "adl")].groupby("path")["stream_id"].nunique().describe().to_dict()
    )
    eda.fig_event_durations(events, figures / "eda_fall_durations.png")
    eda.fig_event_categories(events, figures / "eda_fall_categories.png")
    eda.fig_composition(segments, figures / "eda_composition.png")
    eda.fig_source_reuse(segments, figures / "eda_source_reuse.png")

    train_events = events[eda.split_of(events["stream_id"]) == "train"]
    first = manifest[manifest["stream_id"] == sample[0]].iloc[0]
    candidates = train_events[(train_events["stream_id"] == sample[0]) & ~train_events["flagged"]]
    example = candidates.iloc[(candidates["t_end"] - candidates["t_start"] - lengths.median()).abs().argsort().iloc[0]]
    eda.fig_range_time(first, example, rate, float(get(cfg, "preprocessing.background_tau_seconds")), figures / "eda_range_time_fall.png")
    values, _valid, _ = cached_frame_series(
        Path(first.file_path), rate, float(get(cfg, "preprocessing.background_tau_seconds")), int(get(cfg, "data.chunk_frames")), int(first.n_frames),
        ("bg_energy", "diff_energy"), Path(get(cfg, "experiment.cache_dir")) / "frame_series",
    )
    t0 = max(0.0, float(example.t_start) - 60.0)
    eda.fig_energy_timeline(values, rate, events, segments, sample[0], t0, t0 + 120.0, figures / "eda_energy_timeline.png")
    stats["example_fall"] = {"stream_id": sample[0], "event_id": example.event_id, "t_start": float(example.t_start), "t_end": float(example.t_end)}

    stats["window_length_sensitivity"] = eda.window_label_sensitivity(manifest, events, cfg).to_dict(orient="records")

    windows, matrix, names = eda.train_windows(cfg, manifest, events)
    labels = windows["label"].to_numpy()
    stats["train_windows"] = {"total": int(len(labels)), "positive": int((labels == 1).sum()), "negative": int((labels == 0).sum()), "ambiguous": int((labels < 0).sum())}
    auc = eda.univariate_auc(matrix, labels, names)
    stats["feature_auc_top"] = auc.head(25).round(4).to_dict(orient="records")
    stats["feature_auc_bottom"] = auc.tail(5).round(4).to_dict(orient="records")
    eda.fig_feature_auc(auc, figures / "eda_feature_auc.png")
    junction, samples = eda.junction_table(windows, matrix, names, segments)
    stats["junction_check"] = junction.round(4).to_dict(orient="records")
    eda.fig_junction(samples, figures / "eda_junctions.png")

    stats["background_tau_effect"] = eda.tau_effect(cfg, manifest, events, sample).round(4).to_dict(orient="records")
    kinds, duplicates = eda.duplicate_frame_table(cfg, manifest, events, segments, sample)
    stats["duplicate_frames_by_kind"] = kinds.round(4).to_dict(orient="records")
    stats["duplicate_frames_window_auc"] = duplicates.round(4).to_dict(orient="records")
    location, location_samples = eda.location_table(cfg, manifest, events, sample)
    stats["location_cue"] = location.round(4).to_dict(orient="records")
    eda.fig_location(location_samples, figures / "eda_location_cue.png")

    if args.inspection:
        inspection = json.loads(Path(args.inspection).read_text(encoding="utf-8"))
        stats["inspection"] = {
            "issues": inspection["issues"],
            "nonfinite_total": sum(s.get("values", {}).get("nonfinite_values", 0) for s in inspection["streams"]),
            "per_radar_first_stream": inspection["streams"][0].get("values", {}).get("per_radar"),
            "documents": inspection["documents"],
        }
    write_json(Path(args.stats), stats)
    print(f"figures -> {figures.as_posix()}, numbers -> {args.stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
