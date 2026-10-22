"""EDA and preprocessing analysis used by the NV2 report and notebook.

Descriptive counts cover train and val. Signal and feature analysis uses the
official train split only, so the holdout does not steer modelling choices.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from .config import get  # noqa: E402
from .data_io import read_cir_slice, read_timestamps  # noqa: E402
from .features import window_features  # noqa: E402
from .preprocessing import CausalBackgroundRemover, cached_frame_series  # noqa: E402
from .windowing import POSITIVE, NEGATIVE, WindowSpec, label_windows, make_windows  # noqa: E402

LOG = logging.getLogger(__name__)
SPLIT_COLORS = {"train": "#3b6ea8", "val": "#d9822b"}


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def split_of(stream_ids: pd.Series) -> pd.Series:
    return stream_ids.str.rsplit("_", n=1).str[0]


# ----------------------------------------------------------------------------- tables


def overview_table(manifest: pd.DataFrame, events: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    events = events.assign(split=split_of(events["stream_id"]), duration=events["t_end"] - events["t_start"])
    seconds = segments.assign(seconds=segments["t_end"] - segments["t_start"]).pivot_table(
        index="official_split", columns="kind", values="seconds", aggfunc="sum"
    )
    rows = []
    for split, group in manifest.groupby("official_split"):
        ev = events[events["split"] == split]
        hours = group["duration_s"].sum() / 3600
        rows.append(
            {
                "split": split,
                "streams": len(group),
                "hours": round(hours, 2),
                "falls": len(ev),
                "falls_per_hour": round(len(ev) / hours, 1),
                "flagged_falls": int(ev["flagged"].sum()),
                "fall_time_pct": round(100 * ev["duration"].sum() / (hours * 3600), 2),
                "adl_hours": round(seconds.loc[split].get("adl", 0) / 3600, 2),
                "fall_segment_hours": round(seconds.loc[split].get("fall", 0) / 3600, 2),
                "other_hours": round(seconds.loc[split].get("other", 0) / 3600, 2),
                "subject_groups": int(segments.loc[segments["official_split"] == split, "subject_group"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def window_label_sensitivity(manifest: pd.DataFrame, events: pd.DataFrame, cfg: dict, durations_s=(0.5, 1.0, 2.0, 4.0)) -> pd.DataFrame:
    """Positive/negative/ambiguous window counts on train for several window lengths."""
    rate = float(get(cfg, "data.rate_hz"))
    stride = float(get(cfg, "windowing.stride_seconds"))
    pos = float(get(cfg, "windowing.positive_overlap_threshold"))
    rows = []
    train = manifest[manifest["official_split"] == "train"]
    for duration in durations_s:
        spec = WindowSpec(duration, stride, rate)
        counts = {POSITIVE: 0, NEGATIVE: 0, -1: 0}
        detected = 0
        total = 0
        for row in train.itertuples(index=False):
            t = np.arange(row.n_frames) / rate
            windows = make_windows(row.stream_id, t, np.ones(row.n_frames, bool), spec)
            ev = events[events["stream_id"] == row.stream_id]
            labels, _cov = label_windows(windows["t_start"].to_numpy(), windows["t_end"].to_numpy(), ev, pos)
            for key in counts:
                counts[key] += int((labels == key).sum())
            for e in ev.itertuples(index=False):
                total += 1
                inside = (windows["t_start"] < e.t_end) & (windows["t_end"] > e.t_start)
                detected += bool(np.any(labels[inside.to_numpy()] == POSITIVE))
        rows.append(
            {
                "window_s": duration,
                "positive": counts[POSITIVE],
                "negative": counts[NEGATIVE],
                "ambiguous": counts[-1],
                "positive_pct": round(100 * counts[POSITIVE] / sum(counts.values()), 2),
                "falls_with_a_positive_window_pct": round(100 * detected / total, 1),
            }
        )
    table = pd.DataFrame(rows)
    # tIoU when one window is reported as one event and aligned on a median-length fall
    # (the official 4 s logistic baseline works this way; runs of windows do better)
    median = float(np.median((events["t_end"] - events["t_start"]).to_numpy()))
    table["single_window_event_tiou_at_median_fall"] = [round(min(median, d) / max(median, d), 2) for d in durations_s]
    return table


# ----------------------------------------------------------------------------- figures


def fig_event_durations(events: pd.DataFrame, path: Path) -> Path:
    ev = events.assign(split=split_of(events["stream_id"]), duration=events["t_end"] - events["t_start"])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bins = np.arange(0, 5.25, 0.25)
    for split, group in ev.groupby("split"):
        ax.hist(group["duration"], bins=bins, alpha=0.6, label=f"{split} (n={len(group)})", color=SPLIT_COLORS.get(split))
    ax.axvline(ev["duration"].median(), color="k", ls="--", lw=1, label=f"median {ev['duration'].median():.2f} s")
    ax.set_xlabel("fall duration (s)")
    ax.set_ylabel("events")
    ax.set_title("Annotated fall durations")
    ax.legend()
    return _save(fig, path)


def fig_event_categories(events: pd.DataFrame, path: Path) -> Path:
    ev = events.assign(split=split_of(events["stream_id"]))
    ev["flag"] = np.select([ev["quality"].eq("low") & ev["needs_review"], ev["quality"].eq("low"), ev["needs_review"]], ["low + review", "low quality", "needs review"], "normal")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, column, title in zip(axes, ("fall_type", "dataset", "flag"), ("fall type", "mat position (Dataset)", "quality flags")):
        table = ev.groupby([column, "split"]).size().unstack(fill_value=0)
        table.plot.bar(ax=ax, color=[SPLIT_COLORS.get(c) for c in table.columns], rot=20)
        ax.set_title(title)
        ax.set_xlabel("")
    axes[0].set_ylabel("events")
    return _save(fig, path)


def fig_composition(segments: pd.DataFrame, path: Path) -> Path:
    seg = segments.assign(seconds=segments["t_end"] - segments["t_start"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1, 2.4]})
    kinds = seg.pivot_table(index="official_split", columns="kind", values="seconds", aggfunc="sum") / 3600
    kinds.plot.bar(ax=axes[0], stacked=True, rot=0, color=["#7aa974", "#c45a5a", "#999999"])
    axes[0].set_ylabel("hours")
    axes[0].set_title("Stream time by segment kind")
    adl = seg[seg["kind"] == "adl"].pivot_table(index="label", columns="official_split", values="seconds", aggfunc="sum") / 60
    adl.sort_values("train").plot.barh(ax=axes[1], color=[SPLIT_COLORS.get(c) for c in adl.columns])
    axes[1].set_xlabel("minutes")
    axes[1].set_ylabel("")
    axes[1].set_title("Activities of daily living (ADL)")
    return _save(fig, path)


def fig_source_reuse(segments: pd.DataFrame, path: Path) -> Path:
    train = segments[segments["official_split"] == "train"]
    adl = train[train["kind"] == "adl"].groupby("path")["stream_id"].nunique()
    people = train[train["kind"] == "fall"].groupby("subject_group")["stream_id"].nunique()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    axes[0].hist(adl, bins=np.arange(0.5, adl.max() + 1.5), color="#7aa974")
    axes[0].set_xlabel("number of train streams")
    axes[0].set_ylabel("ADL source files")
    axes[0].set_title("One ADL recording feeds many streams")
    people.sort_values().plot.barh(ax=axes[1], color="#c45a5a")
    axes[1].set_xlabel("number of train streams")
    axes[1].set_title("Train streams containing each fall person")
    return _save(fig, path)


def fig_range_time(row, event, rate: float, tau: float, path: Path, before: float = 8.0, after: float = 8.0) -> Path:
    """Raw CIR and background-removed residual around one fall, on the radar where it is strongest."""
    warmup = 30.0
    first = int(max(0.0, event.t_start - before - warmup) * rate)
    start = int(max(0.0, event.t_start - before) * rate)
    stop = int(min(row.n_frames, (event.t_end + after) * rate))
    block = read_cir_slice(Path(row.file_path), first, stop)
    residual = CausalBackgroundRemover(tau, rate).process(block)[:, start - first :, :]
    raw = block[:, start - first :, :]
    radar = int(np.argmax((residual**2).sum(axis=(1, 2))))
    extent = [start / rate, stop / rate, 0, block.shape[2]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    for ax, data, title in ((axes[0], raw[radar], "raw CIR"), (axes[1], np.abs(residual[radar]), f"|residual| after EMA background (tau={tau:g} s)")):
        limit = np.percentile(np.abs(data), 99)
        ax.imshow(data.T, aspect="auto", origin="lower", extent=extent, cmap="RdBu_r" if ax is axes[0] else "magma",
                  vmin=-limit if ax is axes[0] else 0, vmax=limit)
        for t in (event.t_start, event.t_end):
            ax.axvline(t, color="lime", lw=1.2, ls="--")
        ax.set_title(f"{title}, radar {radar}")
        ax.set_xlabel("stream time (s)")
    axes[0].set_ylabel("range bin")
    fig.suptitle(f"{row.stream_id}: {event.fall_type}, {event.dataset}, {event.t_end - event.t_start:.2f} s (dashed = annotated fall)", fontsize=10)
    return _save(fig, path)


def fig_energy_timeline(values: dict, rate: float, events: pd.DataFrame, segments: pd.DataFrame, stream_id: str, t0: float, t1: float, path: Path) -> Path:
    a, b = int(t0 * rate), int(t1 * rate)
    t = np.arange(a, b) / rate
    fig, axes = plt.subplots(2, 1, figsize=(12, 4.6), sharex=True)
    for ax, name in zip(axes, ("bg_energy", "diff_energy")):
        for radar in range(values[name].shape[1]):
            ax.plot(t, np.log10(values[name][a:b, radar] + 1e-12), lw=0.7, label=f"radar {radar}")
        for e in events[(events["stream_id"] == stream_id) & (events["t_end"] > t0) & (events["t_start"] < t1)].itertuples():
            ax.axvspan(e.t_start, e.t_end, color="red", alpha=0.25)
        for j in segments[(segments["stream_id"] == stream_id) & (segments["t_start"] > t0) & (segments["t_start"] < t1)]["t_start"]:
            ax.axvline(j, color="grey", lw=0.6, ls=":")
        ax.set_ylabel(f"log10 {name}")
    axes[0].legend(loc="upper right", fontsize=8, ncol=3)
    axes[0].set_title(f"{stream_id}: per-frame energies (red = fall, dotted = segment join)")
    axes[1].set_xlabel("stream time (s)")
    return _save(fig, path)


# ----------------------------------------------------------------------------- train-only analysis


def train_windows(cfg: dict, manifest: pd.DataFrame, events: pd.DataFrame, streams: list[str] | None = None, series=("bg_energy", "diff_energy")):
    """Windows, labels and features of train streams, from the frame-series cache."""
    rate = float(get(cfg, "data.rate_hz"))
    tau = float(get(cfg, "preprocessing.background_tau_seconds"))
    spec = WindowSpec(float(get(cfg, "windowing.duration_seconds")), float(get(cfg, "windowing.stride_seconds")), rate)
    cache = Path(get(cfg, "experiment.cache_dir")) / "frame_series" if get(cfg, "experiment.cache_dir") else None
    rows = manifest[manifest["official_split"] == "train"]
    if streams:
        rows = rows[rows["stream_id"].isin(streams)]
    tables, matrices, names = [], [], None
    for row in rows.itertuples(index=False):
        # stop=n_frames gives the same cache key as the pipeline's full-length runs
        values, valid, _ = cached_frame_series(Path(row.file_path), rate, tau, int(get(cfg, "data.chunk_frames")), row.n_frames, tuple(series), cache)
        windows = make_windows(row.stream_id, read_timestamps(Path(row.file_path)), valid, spec)
        matrix, names = window_features(
            values, windows["f_start"].to_numpy(), spec.frames, list(series), list(get(cfg, "features.enabled_features")), rate,
            context_seconds=get(cfg, "features.context_seconds"),
        )
        labels, coverage = label_windows(windows["t_start"].to_numpy(), windows["t_end"].to_numpy(), events[events["stream_id"] == row.stream_id], float(get(cfg, "windowing.positive_overlap_threshold")))
        windows["label"], windows["coverage"] = labels, coverage
        tables.append(windows)
        matrices.append(matrix)
    return pd.concat(tables, ignore_index=True), np.vstack(matrices), names


def univariate_auc(features: np.ndarray, labels: np.ndarray, names: list[str]) -> pd.DataFrame:
    keep = labels >= 0
    rows = []
    for index, name in enumerate(names):
        column = features[keep, index]
        good = np.isfinite(column)
        auc = roc_auc_score(labels[keep][good], column[good]) if len(np.unique(labels[keep][good])) == 2 else np.nan
        rows.append({"feature": name, "auc": auc, "separation": abs(auc - 0.5) if np.isfinite(auc) else np.nan})
    return pd.DataFrame(rows).sort_values("separation", ascending=False).reset_index(drop=True)


def fig_feature_auc(table: pd.DataFrame, path: Path, top: int = 20) -> Path:
    head = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.28 * top + 1))
    ax.barh(head["feature"], head["auc"], color=np.where(head["auc"] >= 0.5, "#3b6ea8", "#c45a5a"))
    ax.axvline(0.5, color="k", lw=0.8)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("ROC-AUC of the single feature (fall vs no-fall windows, train)")
    ax.set_title(f"Top {top} features by univariate separation")
    return _save(fig, path)


def tau_effect(cfg: dict, manifest: pd.DataFrame, events: pd.DataFrame, streams: list[str], taus=(1.0, 3.0, 8.0)) -> pd.DataFrame:
    """Window-level AUC of the mean log residual energy for several background time constants."""
    rows = []
    for tau in taus:
        local = {**cfg, "preprocessing": {**cfg["preprocessing"], "background_tau_seconds": tau}}
        windows, matrix, names = train_windows(local, manifest, events, streams, series=("bg_energy",))
        keep = windows["label"].to_numpy() >= 0
        labels = windows["label"].to_numpy()[keep]
        for feature in ("bg_energy_radarmax_mean", "bg_energy_radarmax_max"):
            column = matrix[keep, names.index(feature)]
            rows.append({"tau_s": tau, "feature": feature, "auc": float(roc_auc_score(labels, column))})
    return pd.DataFrame(rows)


def junction_table(windows: pd.DataFrame, matrix: np.ndarray, names: list[str], segments: pd.DataFrame) -> pd.DataFrame:
    """Peak frame-difference energy in windows that contain a join, versus elsewhere."""
    column = matrix[:, names.index("diff_energy_radarmax_max")]
    contains_join = np.zeros(len(windows), dtype=bool)
    for stream_id, positions in windows.groupby("stream_id").indices.items():
        seg = segments[segments["stream_id"] == stream_id].sort_values("t_start")
        joins = seg["t_start"].to_numpy()[1:]
        kinds = list(zip(seg["kind"].to_numpy()[:-1], seg["kind"].to_numpy()[1:]))
        joins = np.array([j for j, (a, b) in zip(joins, kinds) if a != "fall" and b != "fall"])
        part = windows.iloc[positions]
        if len(joins):
            starts = part["t_start"].to_numpy()[:, None]
            ends = part["t_end"].to_numpy()[:, None]
            contains_join[positions] = np.any((joins[None, :] >= starts) & (joins[None, :] < ends), axis=1)
    labels = windows["label"].to_numpy()
    groups = {
        "no-fall, no join": (labels == NEGATIVE) & ~contains_join,
        "no-fall, contains a non-fall join": (labels == NEGATIVE) & contains_join,
        "fall (positive)": labels == POSITIVE,
    }
    rows = []
    for name, mask in groups.items():
        values = column[mask]
        rows.append({"group": name, "windows": int(mask.sum()), "median": float(np.median(values)), "q90": float(np.quantile(values, 0.9))})
    return pd.DataFrame(rows), {name: column[mask] for name, mask in groups.items()}


def fig_junction(samples: dict, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.boxplot(list(samples.values()), showfliers=False)
    ax.set_xticks(range(1, len(samples) + 1), list(samples.keys()), fontsize=8)
    ax.set_ylabel("max log10 diff_energy (window)")
    ax.set_title("Do segment joins look like falls? (train)")
    return _save(fig, path)


def location_table(cfg: dict, manifest: pd.DataFrame, events: pd.DataFrame, streams: list[str]) -> tuple[pd.DataFrame, dict]:
    """Motion range centroid in fall windows vs no-fall windows: a location cue, not a fall cue."""
    windows, matrix, names = train_windows(cfg, manifest, events, streams, series=("range_centroid",))
    labels = windows["label"].to_numpy()
    rows, samples = [], {}
    for radar in range(3):
        column = matrix[:, names.index(f"range_centroid_r{radar}_median")]
        keep = labels >= 0
        auc = roc_auc_score(labels[keep], column[keep])
        rows.append({"radar": radar, "auc_fall_vs_nofall": float(auc), "median_bin_fall": float(np.median(column[labels == POSITIVE])), "median_bin_nofall": float(np.median(column[labels == NEGATIVE]))})
        samples[radar] = (column[labels == POSITIVE], column[labels == NEGATIVE])
    return pd.DataFrame(rows), samples


def duplicate_frame_table(cfg: dict, manifest: pd.DataFrame, events: pd.DataFrame, segments: pd.DataFrame, streams: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frames identical to the previous frame (nearest-frame alignment to 50 Hz).

    Returns per-kind duplicate fractions and the window-level AUC of the duplicate
    fraction for fall vs no-fall windows. A high AUC would mean the radar frame rate
    of the source recording betrays the segment kind (a shortcut, not a fall cue).
    """
    from .data_io import iter_cir_chunks

    rate = float(get(cfg, "data.rate_hz"))
    spec = WindowSpec(float(get(cfg, "windowing.duration_seconds")), float(get(cfg, "windowing.stride_seconds")), rate)
    kind_rows, window_rows = [], []
    for row in manifest[manifest["stream_id"].isin(streams)].itertuples(index=False):
        flags, previous = [], None
        for _start, block in iter_cir_chunks(Path(row.file_path), int(get(cfg, "data.chunk_frames"))):
            head = block[:, :1, :] if previous is None else previous
            joined = np.concatenate([head, block], axis=1)
            flags.append(np.all(joined[:, 1:, :] == joined[:, :-1, :], axis=2).T)  # (frames, radars)
            previous = block[:, -1:, :]
        dup = np.concatenate(flags).astype(np.float64)
        dup[0] = 0.0
        times = np.arange(len(dup)) / rate
        for seg in segments[segments["stream_id"] == row.stream_id].itertuples(index=False):
            inside = (times >= seg.t_start + 0.3) & (times < seg.t_end - 0.3)
            if inside.sum() >= 50:
                kind_rows.append({"kind": seg.kind, "dup_fraction": float(dup[inside].mean())})
        windows = make_windows(row.stream_id, times, np.ones(len(dup), bool), spec)
        labels, _ = label_windows(windows["t_start"].to_numpy(), windows["t_end"].to_numpy(), events[events["stream_id"] == row.stream_id], float(get(cfg, "windowing.positive_overlap_threshold")))
        cumulative = np.vstack([np.zeros((1, dup.shape[1])), np.cumsum(dup, axis=0)])
        frac = (cumulative[windows["f_stop"].to_numpy()] - cumulative[windows["f_start"].to_numpy()]).mean(axis=1) / spec.frames
        window_rows.append(pd.DataFrame({"label": labels, "dup_fraction": frac}))
    kinds = pd.DataFrame(kind_rows).groupby("kind")["dup_fraction"].describe()[["count", "mean", "min", "50%", "max"]]
    kinds["implied_native_rate_hz"] = rate * (1 - kinds["mean"])
    win = pd.concat(window_rows, ignore_index=True)
    win = win[win["label"] >= 0]
    auc = roc_auc_score(win["label"], win["dup_fraction"])
    summary = pd.DataFrame([{"windows": len(win), "auc_dup_fraction_fall_vs_nofall": float(auc), "median_fall": float(win.loc[win.label == 1, "dup_fraction"].median()), "median_nofall": float(win.loc[win.label == 0, "dup_fraction"].median())}])
    return kinds.reset_index(), summary


def fig_location(samples: dict, path: Path) -> Path:
    fig, axes = plt.subplots(1, len(samples), figsize=(12, 3.2), sharey=True)
    bins = np.linspace(0, 368, 47)
    for ax, (radar, (fall, other)) in zip(axes, samples.items()):
        ax.hist(other, bins=bins, density=True, alpha=0.6, label="no fall", color="#7aa974")
        ax.hist(fall, bins=bins, density=True, alpha=0.6, label="fall", color="#c45a5a")
        ax.set_title(f"radar {radar}")
        ax.set_xlabel("motion range centroid (bin)")
    axes[0].set_ylabel("density")
    axes[0].legend()
    fig.suptitle("Where the motion is: fall windows sit at fixed mat distances (train)", fontsize=10)
    return _save(fig, path)
