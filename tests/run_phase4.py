"""Run Phase 4 Step 0 — hạ tầng đánh giá.

Usage:
    python experiments/run_phase4.py --step 0          # Run Step 0 (0.1–0.6)
    python experiments/run_phase4.py --step 0 --fast   # Skip leakage probe (faster)

Step 0 sub-steps:
  0.1  OOF harness (build_folds, verify subject exclusion)
  0.2  pipeline_19c (OOF threshold sweep + val eval)
  0.3  Physics checks
  0.4  Join detector
  0.5  Leakage probe
  0.6  Error audit + gates

Ends with CHECKPOINT report. Does NOT run pipeline_21.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "experiments" / "audit" / "reports"


def step0(args):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GĐ4 STEP 0: Hạ tầng đánh giá")
    print("=" * 70)

    # ── 0.1 OOF harness ──────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Step 0.1: OOF harness — Build folds")
    print("─" * 60)
    from experiments.audit.oof_harness import build_folds, load_composition
    comp_rows = load_composition()
    print(f"Composition CSV: {len(comp_rows)} segments")
    cols = list(comp_rows[0].keys()) if comp_rows else []
    print(f"Columns: {cols}")

    folds = build_folds()
    fold_summary = [{"test": test, "n_train": len(train)} for test, train in folds]
    print(f"\nFold summary ({len(folds)} folds):")
    for f in fold_summary:
        print(f"  test={f['test']}  n_train={f['n_train']}")

    fold_report = REPORTS_DIR / "fold_summary.json"
    fold_report.write_text(json.dumps(fold_summary, indent=2), encoding="utf-8")
    print(f"Fold summary saved: {fold_report}")

    # ── 0.3 Physics checks (early — needed for 0.4 envelope) ─────────────────
    print("\n" + "─" * 60)
    print("Step 0.3: Physics checks")
    print("─" * 60)
    from experiments.audit.physics_checks import run_physics_checks
    phys = run_physics_checks(n_streams=3)
    axis_layout = phys.get("axis_layout", "iq_concat_184")
    radar_sync  = phys.get("radar_sync", True)
    print(f"\n→ axis_layout = {axis_layout}")
    print(f"→ radar_sync  = {radar_sync}")
    if axis_layout == "unclear":
        print("\n🚨 BÁNG ĐỘNG ĐỎ: axis_layout = unclear!")

    # ── 0.4 Join detector ────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Step 0.4: Join detector")
    print("─" * 60)
    from experiments.preprocessing.join_detector import evaluate_join_detector
    join_report = evaluate_join_detector(threshold=3.0, n_streams=13)
    join_detector_ok = join_report.get("join_detector_ok", False)
    print(f"\n→ join_detector_ok = {join_detector_ok}")
    print(f"  recall={join_report['recall']:.4f}  precision={join_report['precision']:.4f}")

    # ── 0.2 pipeline_19c (most expensive step) ───────────────────────────────
    print("\n" + "─" * 60)
    print("Step 0.2: pipeline_19c — OOF threshold sweep + val")
    print("─" * 60)
    if args.fast:
        print("  [FAST mode] Skipping OOF threshold sweep, using threshold=0.25")
        import subprocess
        result = subprocess.run(
            [sys.executable, str(ROOT / "experiments" / "pipelines" / "pipeline_19c.py"),
             "--skip-oof"],
            cwd=str(ROOT), capture_output=False,
        )
    else:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(ROOT / "experiments" / "pipelines" / "pipeline_19c.py")],
            cwd=str(ROOT), capture_output=False,
        )

    # Read log to get results
    from experiments.utils import LOG_FILE
    import json as _json
    log = _json.loads(LOG_FILE.read_text(encoding="utf-8"))
    p19c_entry = next((e for e in log if e["experiment_id"] == "pipeline_19c"), None)

    # ── 0.5 Leakage probe ────────────────────────────────────────────────────
    if not args.skip_probe:
        print("\n" + "─" * 60)
        print("Step 0.5: Leakage probe")
        print("─" * 60)
        from experiments.audit.leakage_probe import run_leakage_probe
        probe_results = run_leakage_probe(axis_layout=axis_layout)
    else:
        print("\n  [Skipping leakage probe (--skip-probe)]")
        probe_results = {}

    # ── 0.6 Error audit + gates ──────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Step 0.6: Error audit")
    print("─" * 60)
    oof_csv = ROOT / "oof_pipeline_19c.csv"
    if oof_csv.exists():
        from experiments.audit.error_audit import run_error_audit
        gates = run_error_audit(oof_csv=oof_csv)
    else:
        print(f"  OOF CSV not found at {oof_csv}")
        print("  Skipping error audit (run pipeline_19c first)")
        gates = {"error": "OOF CSV not found"}

    # ── CHECKPOINT REPORT ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CHECKPOINT REPORT — GĐ4 STEP 0")
    print("=" * 70)

    print("\n📋 OOF Harness (Step 0.1)")
    print(f"  Folds: {len(folds)} LOSO folds on train_00–12")
    min_train = min(f['n_train'] for f in fold_summary)
    max_train = max(f['n_train'] for f in fold_summary)
    print(f"  Train sizes: min={min_train}, max={max_train}")
    has_exclusion = any(f['n_train'] < 12 for f in fold_summary)
    print(f"  Subject exclusion active: {has_exclusion}")
    print(f"  Composition columns: {cols}")

    print("\n📋 pipeline_19c (Step 0.2) — NEW BASELINE R")
    if p19c_entry:
        print(f"  OOF F1_mean = {p19c_entry.get('oof_f1_mean')} (= R)")
        print(f"  OOF F1_01  = {p19c_entry.get('oof_f1_01')}")
        print(f"  OOF F1_03  = {p19c_entry.get('oof_f1_03')}")
        print(f"  OOF F1_05  = {p19c_entry.get('oof_f1_05')}")
        print(f"  Val F1_mean = {p19c_entry.get('f1_mean')}")
        print(f"  Val F1_05  = {p19c_entry.get('f1_05')}")
        print(f"  Decision: {p19c_entry.get('decision')}")
        print(f"  Notes: {p19c_entry.get('notes','')}")
    else:
        print("  pipeline_19c entry not found in log")

    print("\n📋 Physics Checks (Step 0.3)")
    print(f"  axis_layout     = {axis_layout}")
    print(f"  radar_sync      = {radar_sync}")
    print(f"  native_rate_hz  = {phys.get('native_rate_hz_estimate')}")
    print(f"  has_negatives   = {phys.get('has_negative_values')}  ({phys.get('negative_fraction', 0):.3%})")
    print(f"  corr_halves     = {phys.get('corr_halves')}")
    print(f"  corr_even_odd   = {phys.get('corr_even_odd')}")
    if axis_layout == "unclear":
        print("  🚨 CẢNH BÁO ĐỎ: axis_layout = unclear!")

    print("\n📋 Join Detector (Step 0.4)")
    print(f"  join_detector_ok = {join_detector_ok}")
    print(f"  recall    = {join_report['recall']:.4f}")
    print(f"  precision = {join_report['precision']:.4f}")
    print(f"  TP={join_report['tp']}  FP={join_report['fp']}  FN={join_report['fn']}")

    print("\n📋 Leakage Probe (Step 0.5)")
    if probe_results and "error" not in probe_results:
        print(f"  Probe 1 AUC (all 27f): {probe_results.get('probe1_auc_all')} "
              f"{'❌ FAIL' if probe_results.get('probe1_fail') else '✅ PASS'} (thr=0.65)")
        print(f"  Probe 2 macro-F1: {probe_results.get('probe2_macro_f1')} "
              f"{'❌ FAIL' if probe_results.get('probe2_fail') else '✅ PASS'} (thr=0.45)")
        print(f"  Probe score AUC: {probe_results.get('probe_score_auc')}")
        print("  By feature group:")
        for gname, auc in probe_results.get("probe1_by_group", {}).items():
            flag = " ⚠️ HIGH" if auc > 0.9 else ""
            print(f"    {gname}: {auc}{flag}")
        if probe_results.get("fails"):
            print("\n  🚨 FAILURES:")
            for fail in probe_results["fails"]:
                print(f"    {fail}")
    else:
        print(f"  {'Skipped' if args.skip_probe else probe_results.get('error', 'Unknown error')}")

    print("\n📋 Error Audit + Gates (Step 0.6)")
    if "error" not in gates:
        print(f"  FN={gates.get('n_fn')}  FP={gates.get('n_fp')}  "
              f"LOW_IOU={gates.get('n_low_iou')}")
        print(f"  gate_p22 = {gates.get('gate_p22')} "
              f"({gates.get('gate_p22_frac', 0):.1%} FP after fall, thr≥25%)")
        print(f"  gate_p26 = {gates.get('gate_p26')} "
              f"({gates.get('gate_p26_frac', 0):.1%} FP near join, thr≥25%)")
        print(f"  gate_p27 = {gates.get('gate_p27')} "
              f"({gates.get('gate_p27_frac', 0):.1%} FN no S1, thr≥33%)")
        stats = gates.get("stats", {})
        print(f"  Fall duration p5/p50/p95: {stats.get('fall_dur_p5')}/{stats.get('fall_dur_p50')}/{stats.get('fall_dur_p95')} s")
        print(f"  Min center gap: {stats.get('min_center_gap_s')} s")
        print(f"  Flagged positives: {stats.get('flagged_positive_count')}")
    else:
        print(f"  Error: {gates.get('error')}")

    print("\n" + "─" * 60)
    print("NEXT STEPS (sau khi người dùng xác nhận):")
    print("  - Chạy pipeline_21 với Prompt 4B")
    print("  - Base = pipeline_19c")
    print(f"  - R (OOF F1_mean) = {p19c_entry.get('oof_f1_mean') if p19c_entry else 'N/A'}")
    print("─" * 60)
    print("\n🛑 DỪNG — Chờ người dùng xác nhận checkpoint trước khi chạy pipeline_21")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 4")
    parser.add_argument("--step", type=int, default=0, help="Step to run (0 = Step 0)")
    parser.add_argument("--fast", action="store_true",
                        help="Skip OOF threshold sweep (use thr=0.25)")
    parser.add_argument("--skip-probe", action="store_true",
                        help="Skip leakage probe")
    args = parser.parse_args()

    if args.step == 0:
        step0(args)
    else:
        print(f"Step {args.step} not implemented yet. Run Prompt 4B for pipeline_21+.")


if __name__ == "__main__":
    main()
