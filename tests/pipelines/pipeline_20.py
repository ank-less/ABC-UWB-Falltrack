"""Pipeline 20: p19 + final val-set threshold sweep (val leakage — không nộp bài)."""
from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.utils import load_val, save_sub, run_eval, run_pipeline, SUB_DIR
from experiments.models.lightgbm_classifier import (
    load_classifier_rich_aug, load_classifier_base, find_best_threshold_on_train,
)
from experiments.pipelines.pipeline_19 import run as p19_run

import numpy as np


def run_sweep(streams, model, sweep_thresholds=None):
    """Sweep threshold on val; return (best_threshold, best_rows)."""
    if sweep_thresholds is None:
        sweep_thresholds = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
    best_f1 = 0.0
    best_thr = sweep_thresholds[0]
    best_rows = []
    print("  Val threshold sweep (p20):")
    for t in sweep_thresholds:
        rows = p19_run(streams, model, threshold=t)
        path = save_sub(rows, "p20_sweep_tmp")
        m    = run_eval(path)
        print(f"    t={t:.2f} → F1={m['f1_mean']:.4f} "
              f"TP={m['tp']} FP={m['fp']} FN={m['fn']}", flush=True)
        if m["f1_mean"] > best_f1:
            best_f1   = m["f1_mean"]
            best_thr  = t
            best_rows = rows
    try:
        os.remove(SUB_DIR / "submission_p20_sweep_tmp.csv")
    except Exception:
        pass
    print(f"  Best val threshold: t={best_thr:.2f} (F1={best_f1:.4f})")
    return best_thr, best_rows


def run(streams, model, sweep_thresholds=None):
    """Return rows using best val-sweep threshold."""
    _, rows = run_sweep(streams, model, sweep_thresholds)
    return rows


def main():
    streams    = load_val()
    model_base = load_classifier_base()
    opt_thr    = find_best_threshold_on_train(model_base)
    p19_thr    = float(np.clip(opt_thr, 0.10, 0.25))
    model      = load_classifier_rich_aug()
    best_sweep_thr, best_sweep_rows = run_sweep(
        streams, model, [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
    )
    run_pipeline(20, f"p19 + val_sweep_best_thr={best_sweep_thr:.2f} (rich+aug+dual_gate)", best_sweep_rows, 0.6819)


if __name__ == "__main__":
    main()
