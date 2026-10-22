"""Chạy toàn bộ 21 pipeline theo thứ tự, ghi log vào experiments_log.json.

Chạy: python experiments/run_all.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.utils import load_val, run_pipeline, LOG_FILE
from experiments.models.lightgbm_classifier import (
    load_classifier,
    load_classifier_base,
    load_classifier_augmented,
    load_classifier_rich,
    load_classifier_rich_aug,
    find_best_threshold_on_train,
    window_features,
)
import experiments.pipelines.pipeline_00_baseline as p00_mod
import experiments.pipelines.pipeline_01 as p01_mod
import experiments.pipelines.pipeline_02 as p02_mod
import experiments.pipelines.pipeline_03 as p03_mod
import experiments.pipelines.pipeline_04 as p04_mod
import experiments.pipelines.pipeline_05 as p05_mod
import experiments.pipelines.pipeline_06 as p06_mod
import experiments.pipelines.pipeline_07 as p07_mod
import experiments.pipelines.pipeline_08 as p08_mod
import experiments.pipelines.pipeline_09 as p09_mod
import experiments.pipelines.pipeline_10 as p10_mod
import experiments.pipelines.pipeline_11 as p11_mod
import experiments.pipelines.pipeline_12 as p12_mod
import experiments.pipelines.pipeline_13 as p13_mod
import experiments.pipelines.pipeline_14 as p14_mod
import experiments.pipelines.pipeline_15 as p15_mod
import experiments.pipelines.pipeline_16 as p16_mod
import experiments.pipelines.pipeline_18 as p18_mod
import experiments.pipelines.pipeline_19 as p19_mod
from experiments.pipelines.pipeline_20 import run_sweep as p20_sweep


def main():
    print("=" * 80)
    print("RUN ALL — 21 PIPELINES")
    print("=" * 80)

    streams = load_val()
    print(f"Loaded {len(streams)} val streams\n")

    prev = None

    # ── GIAI ĐOẠN 1: pipeline 00–07 ──────────────────────────────────────────
    print("=== GIAI ĐOẠN 1: Giảm FP (pipeline 00–07) ===\n")

    print("── Pipeline 00: Baseline ──")
    prev = run_pipeline(0, "Baseline gốc: EMA background + energy threshold",
                        p00_mod.run(streams), None)

    print("── Pipeline 01: Baseline + temporal smoothing ──")
    prev = run_pipeline(1, "baseline + Gaussian temporal smoothing (σ=0.3s)",
                        p01_mod.run(streams), prev)

    print("── Pipeline 02: Baseline + smoothing + duration filter ──")
    prev = run_pipeline(2, "baseline + smoothing + duration_filter(>=1.0s)",
                        p02_mod.run(streams), prev)

    print("── Pipeline 03: Smoothing + dur_filter + gap_merger ──")
    prev = run_pipeline(3, "baseline + smoothing + duration_filter + gap_merger(1.5s)",
                        p03_mod.run(streams), prev)

    print("── Pipeline 04: Chebyshev + smoothing ──")
    prev = run_pipeline(4, "Chebyshev bandpass(0.5–15 Hz) + energy threshold + smoothing",
                        p04_mod.run(streams), prev)

    print("── Pipeline 05: Chebyshev + IQR + smoothing ──")
    prev = run_pipeline(5, "Chebyshev + IQR normalization + energy threshold + smoothing",
                        p05_mod.run(streams), prev)

    print("\n── Training classifier for p06–p07 ──")
    model_orig = load_classifier()
    print()

    print("── Pipeline 06: Chebyshev + IQR + classifier ──")
    prev = run_pipeline(6, "Chebyshev + IQR + LightGBM/RF classifier + smoothing",
                        p06_mod.run(streams, model_orig), prev)

    print("── Pipeline 07: Chebyshev + IQR + classifier + dur + gap ──")
    prev = run_pipeline(7, "Chebyshev + IQR + classifier + duration_filter + gap_merger",
                        p07_mod.run(streams, model_orig), prev)

    # ── GIAI ĐOẠN 2: pipeline 08–14 ──────────────────────────────────────────
    print("\n=== GIAI ĐOẠN 2: Cải thiện model (pipeline 08–14) ===\n")

    print("── Pipeline 08: energy candidates → LightGBM gate + postproc ──")
    prev = run_pipeline(8, "p04_base + LightGBM_gate(thresh=0.1) + dur_filter(0.6s) + gap_merge(1.0s)",
                        p08_mod.run(streams, model_orig), prev)

    print("── Pipeline 09: pipeline_08 + boundary refiner ──")
    prev = run_pipeline(9, "p04_base + LightGBM_gate + boundary_refiner(frac=0.4,ctx=2s)",
                        p09_mod.run(streams, model_orig), prev)

    print("── Pipeline 10: per-radar detection + majority-vote fusion ──")
    prev = run_pipeline(10, "3_radar_independent_p04 + majority_vote_fusion(2/3)",
                        p10_mod.run(streams), prev)

    print("── Pipeline 11: two-stage (low threshold + LightGBM confirm) ──")
    prev = run_pipeline(11, "two_stage: energy(pct=95) → LightGBM_confirm(thresh=0.2) + gap_merge(1.5s)",
                        p11_mod.run(streams, model_orig), prev)

    print("── Pipeline 12: 3-state HSMM (fall + lying phase extension) ──")
    prev = run_pipeline(12, "Chebyshev + 3state_HSMM: fall_mask + forward_lying_dilation(2s)",
                        p12_mod.run(streams), prev)

    print("── Pipeline 13: p04 + window shrinker to 1.6s ──")
    prev = run_pipeline(13, "p04_base + window_shrinker(target=1.6s around energy peak)",
                        p13_mod.run(streams), prev)

    print("── Pipeline 14: p04 + Gaussian heatmap boundary ──")
    prev = run_pipeline(14, "p04_base + Gaussian_heatmap_boundary(k=1.2)",
                        p14_mod.run(streams), prev)

    # ── GIAI ĐOẠN 3: pipeline 15–20 ──────────────────────────────────────────
    print("\n=== GIAI ĐOẠN 3: Tối ưu & Ensemble (pipeline 15–20) ===\n")

    print("── Training base classifier (p15, p16) ──")
    model_base = load_classifier_base()
    print()

    print("── Pipeline 15: dual-model validation (p09 + p14 Gaussian gate) ──")
    prev = run_pipeline(15, "p09_base + p14_Gaussian_gate(pct=98,k=1.2): keep if p14_confirms OR lgb>0.30",
                        p15_mod.run(streams, model_base), prev)

    print("\n── Pipeline 16: train-optimised LGB threshold ──")
    opt_thr = find_best_threshold_on_train(model_base)
    prev = run_pipeline(16, f"p09_base + train_opt_threshold={opt_thr:.2f} + boundary_refiner",
                        p16_mod.run(streams, model_base, threshold=opt_thr), prev)

    print("\n── Training augmented LGB (9 features + aug) ──")
    model_aug = load_classifier_augmented(feat_fn=window_features, label="base+aug")
    print()

    print("── Pipeline 17: CutMix/noise-augmented model ──")
    prev = run_pipeline(17, "p09 + feature-aug training (noise+scale+mixup, 9 features, thr=0.10)",
                        p16_mod.run(streams, model_aug, threshold=0.10), prev)

    print("\n── Training rich LGB (27 features) ──")
    model_rich = load_classifier_rich()
    print()

    print("── Pipeline 18: rich features (27) ──")
    prev = run_pipeline(18, "p09 + window_features_rich(27 features): temporal asymmetry, peak timing, cross-radar corr, range ratio",
                        p18_mod.run(streams, model_rich), prev)

    print("\n── Training rich+aug LGB (27 features + augmentation) ──")
    model_rich_aug = load_classifier_rich_aug()
    print()

    p19_thr = float(np.clip(opt_thr, 0.10, 0.25))
    print(f"── Pipeline 19: full combination (rich+aug + CV_thr={p19_thr:.2f} + dual gate) ──")
    prev = run_pipeline(19, f"rich+aug(27f) + CV_thr={p19_thr:.2f} + p14_dual_gate + boundary_refiner",
                        p19_mod.run(streams, model_rich_aug, threshold=p19_thr), prev)

    print("\n── Pipeline 20: final val-set threshold sweep ──")
    best_sweep_thr, best_sweep_rows = p20_sweep(
        streams, model_rich_aug, [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
    )
    prev = run_pipeline(20, f"p19 + val_sweep_best_thr={best_sweep_thr:.2f} (rich+aug+dual_gate)",
                        best_sweep_rows, prev)

    # ── Final report ──────────────────────────────────────────────────────────
    log = json.loads(LOG_FILE.read_text(encoding="utf-8"))
    all_ids = {f"pipeline_{i:02d}" for i in range(21)}
    all_entries = [e for e in log if e["experiment_id"] in all_ids]
    all_entries.sort(key=lambda e: int(e["experiment_id"].split("_")[1]))

    print("\n" + "=" * 90)
    print("BẢNG TỔNG HỢP 21 PIPELINE")
    print("=" * 90)
    print(f"{'ID':<16} {'F1_mean':>8} {'F1_01':>7} {'F1_03':>7} {'F1_05':>7} "
          f"{'TP':>4} {'FP':>4} {'FN':>4} {'FA/h':>6} {'Δ':>8}  Giai đoạn")
    print("-" * 90)
    overall_best_f1 = max(e["f1_mean"] for e in all_entries)
    for e in all_entries:
        pid   = int(e["experiment_id"].split("_")[1])
        phase = "GĐ1" if pid <= 7 else ("GĐ2" if pid <= 14 else "GĐ3")
        marker = " ◄ BEST" if e["f1_mean"] == overall_best_f1 else ""
        print(
            f"{e['experiment_id']:<16} {e['f1_mean']:>8.4f} {e['f1_01']:>7.4f} "
            f"{e['f1_03']:>7.4f} {e['f1_05']:>7.4f} {e['tp']:>4} {e['fp']:>4} "
            f"{e['fn']:>4} {e['false_alarms_per_hour']:>6.1f} "
            f"{e.get('delta_f1', 0):>+8.4f}  {phase}{marker}"
        )
    print("=" * 90)

    global_best = max(all_entries, key=lambda e: e["f1_mean"])
    print(f"\nPipeline tốt nhất: {global_best['experiment_id']}  (F1_mean = {global_best['f1_mean']:.4f})")
    if global_best["f1_mean"] >= 0.6:
        print("✓ ĐẠT MỤC TIÊU GĐ3: F1 mean ≥ 0.6")
    elif global_best["f1_mean"] >= 0.5:
        print("✓ Duy trì mục tiêu GĐ2 (F1 > 0.5), chưa đạt GĐ3 (F1 > 0.6)")
    else:
        print("✗ Chưa đạt F1 > 0.5")
    print("=" * 90)


if __name__ == "__main__":
    main()
