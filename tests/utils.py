"""Shared helpers for all experiment pipelines."""
from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from abc2027.baseline import (
    STEP_S, WINDOW, HOP,
    frame_energy, iter_windows, runs_to_events, window_features,
)
from abc2027.evaluate import (
    evaluate, load_ground_truth, stream_durations,
    temporal_iou, validate_predictions,
)

STREAMS_VAL   = ROOT / "streams" / "val"
STREAMS_TRAIN = ROOT / "streams" / "train"
EVENTS_CSV    = ROOT / "streams_events.csv"
LOG_FILE      = ROOT / "experiments_log.json"
SUB_DIR       = ROOT / "submissions"
SUB_DIR.mkdir(exist_ok=True)
RATE_HZ       = 50.0


def load_val():
    out = {}
    for p in sorted(STREAMS_VAL.glob("stream_*.h5")):
        sid = f"val_{int(p.stem.split('_')[1]):02d}"
        with h5py.File(p) as f:
            out[sid] = f["cir"][:]
    return out


def save_sub(rows, name):
    path = SUB_DIR / f"submission_{name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stream_id", "t_start", "t_end", "score", "t_alarm"])
        w.writeheader()
        w.writerows(rows)
    return path


def run_eval(pred_path):
    durs  = stream_durations(STREAMS_VAL)
    preds = validate_predictions(pred_path, durs)
    gt    = load_ground_truth(EVENTS_CSV, ignore_flagged=True)
    result = evaluate(gt, preds, durs)
    rs = result["ranking_score"]
    bt = result["by_theta"]
    return {
        "f1_mean": round(rs["f1_mean"], 4),
        "f1_01":  round(bt["0.1"]["f1"], 4),
        "f1_03":  round(bt["0.3"]["f1"], 4),
        "f1_05":  round(bt["0.5"]["f1"], 4),
        "tp":     bt["0.3"]["tp"],
        "fp":     bt["0.3"]["fp"],
        "fn":     bt["0.3"]["fn"],
        "false_alarms_per_hour": round(rs["false_alarms_per_hour"], 1),
    }


def append_log(entry):
    log: list = []
    if LOG_FILE.exists():
        log = json.loads(LOG_FILE.read_text(encoding="utf-8"))
    ids = [e["experiment_id"] for e in log]
    if entry["experiment_id"] in ids:
        log[ids.index(entry["experiment_id"])] = entry
    else:
        log.append(entry)
    LOG_FILE.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")


def make_rows(sid, events):
    return [
        {
            "stream_id": sid,
            "t_start":  f"{e['t_start']:.3f}",
            "t_end":    f"{e['t_end']:.3f}",
            "score":    f"{e['score']:.6f}",
            "t_alarm":  f"{e['t_alarm']:.3f}",
        }
        for e in events
    ]


def run_pipeline(pid, desc, rows, prev_f1=None):
    name = f"pipeline_{pid:02d}"
    path = save_sub(rows, name)
    m    = run_eval(path)
    delta = round(m["f1_mean"] - prev_f1, 4) if prev_f1 is not None else 0.0
    entry = {
        "experiment_id": name,
        "description": desc,
        "components": [],
        **m,
        "delta_f1": delta,
        "notes": "",
    }
    append_log(entry)
    print(
        f"[{name}] F1_mean={m['f1_mean']:.4f}  "
        f"TP={m['tp']} FP={m['fp']} FN={m['fn']}  "
        f"FA/h={m['false_alarms_per_hour']:.1f}  "
        f"Δ={delta:+.4f}"
    )
    return m["f1_mean"]


def energy_events_raw(energy, pct=98.0, min_dur=0.6, merge_gap=0.8):
    level = float(np.percentile(energy, pct))
    if level <= 0:
        return []
    return runs_to_events(energy >= level, energy / level, min_dur, merge_gap)
