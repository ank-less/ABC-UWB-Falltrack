"""Run all GĐ4 pipelines (21 → 25) in sequence.

Prerequisites:
  - oof_pipeline_19c.csv must exist (run pipeline_19c.py first)
  - experiments/audit/reports/ should have physics_checks.json, join_detector.json

Usage:
    python experiments/run_gd4_pipelines.py [--from 21] [--to 25] [--dry-run]

Each pipeline:
  1. Checks gate (if applicable)
  2. Runs OOF evaluation
  3. Makes KEEP/DROP decision (based on OOF only)
  4. Runs val exactly once (if KEEP)
  5. Logs to experiments_log.json

pipeline_26 is automatically SKIPPED (join_detector_ok = false → gate not met).
pipeline_27 needs gate_p27 from error_audit.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "experiments" / "audit" / "reports"
LOG_FILE    = ROOT / "experiments_log.json"


def get_log_entry(eid: str) -> dict:
    if not LOG_FILE.exists():
        return {}
    log = json.loads(LOG_FILE.read_text(encoding="utf-8"))
    return next((e for e in log if e["experiment_id"] == eid), {})


def check_prerequisites():
    oof_csv = ROOT / "oof_pipeline_19c.csv"
    if not oof_csv.exists():
        print("❌ oof_pipeline_19c.csv not found.")
        print("   Run: python experiments/pipelines/pipeline_19c.py")
        return False

    p19c = get_log_entry("pipeline_19c")
    if not p19c:
        print("❌ pipeline_19c not found in experiments_log.json.")
        return False

    R = p19c.get("oof_f1_mean")
    print(f"✓ Baseline R = {R} (from pipeline_19c OOF)")
    print(f"  Val F1_mean = {p19c.get('f1_mean')}")

    gates_f = REPORTS_DIR / "gates.json"
    if gates_f.exists():
        gates = json.loads(gates_f.read_text(encoding="utf-8"))
        print(f"✓ gates.json: p22={gates.get('gate_p22')} p26={gates.get('gate_p26')} p27={gates.get('gate_p27')}")
    else:
        print("⚠️  gates.json not found — running error audit first…")
        return "need_audit"

    return True


def run_error_audit():
    print("\n=== Running Error Audit (to get gates.json) ===")
    oof_csv = ROOT / "oof_pipeline_19c.csv"
    result = subprocess.run(
        [sys.executable, str(ROOT / "experiments" / "audit" / "error_audit.py"),
         "--oof-csv", str(oof_csv)],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print("⚠️  error_audit.py returned non-zero — continuing anyway")


def run_pipeline(name: str, dry_run: bool = False) -> str:
    """Run a pipeline script, return its decision from log."""
    script = ROOT / "experiments" / "pipelines" / f"{name}.py"
    if not script.exists():
        print(f"❌ {script} not found — skip")
        return "MISSING"

    print(f"\n{'='*70}")
    print(f"Running {name}…")
    print(f"{'='*70}")

    if dry_run:
        print(f"  [DRY RUN] Would run: python {script}")
        return "DRY_RUN"

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print(f"⚠️  {name} exited with code {result.returncode}")

    # Read decision from log
    entry = get_log_entry(name)
    decision = entry.get("decision", "UNKNOWN")
    oof_f1   = entry.get("oof_f1_mean", "?")
    val_f1   = entry.get("f1_mean", "?")
    print(f"\n  → {name}: decision={decision}, OOF F1={oof_f1}, Val F1={val_f1}")
    return decision


def check_gate(pipeline_id: str) -> bool:
    """Check if a gated pipeline should run."""
    gates_f = REPORTS_DIR / "gates.json"
    if not gates_f.exists():
        print(f"  ⚠️  gates.json missing — treating gate as closed for {pipeline_id}")
        return False
    gates = json.loads(gates_f.read_text(encoding="utf-8"))

    if pipeline_id == "pipeline_22":
        ok = gates.get("gate_p22", False)
        print(f"  gate_p22 = {ok}")
        return bool(ok)
    elif pipeline_id == "pipeline_26":
        ok = gates.get("gate_p26", False)
        jd_ok = json.loads(
            (REPORTS_DIR / "join_detector.json").read_text(encoding="utf-8")
        ).get("join_detector_ok", False) if (REPORTS_DIR / "join_detector.json").exists() else False
        print(f"  gate_p26={ok}, join_detector_ok={jd_ok}")
        return bool(ok) and bool(jd_ok)
    elif pipeline_id == "pipeline_27":
        ok = gates.get("gate_p27", False)
        print(f"  gate_p27 = {ok}")
        return bool(ok)
    return True


def main():
    parser = argparse.ArgumentParser(description="Run GĐ4 pipelines 21–25")
    parser.add_argument("--from", dest="from_p", type=int, default=21,
                        help="Start from this pipeline number (default: 21)")
    parser.add_argument("--to",   dest="to_p",   type=int, default=25,
                        help="End at this pipeline number (default: 25)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run without running")
    args = parser.parse_args()

    print("=" * 70)
    print("GĐ4 — Pipelines 21–25 (+ 26/27 if gates open)")
    print("=" * 70)

    status = check_prerequisites()
    if status is False:
        sys.exit(1)
    if status == "need_audit":
        run_error_audit()

    pipelines = [f"pipeline_{i:02d}" for i in range(args.from_p, args.to_p + 1)]
    gated = {"pipeline_22", "pipeline_26", "pipeline_27"}

    drop_count = 0
    MAX_CONSEC_DROP = 3

    for pid in pipelines:
        pid_num = int(pid.split("_")[1])

        # pipeline_26 auto-skipped (join_detector_ok=false)
        if pid_num == 26:
            print(f"\n  {pid}: SKIPPED_GATE (join_detector_ok=false — no join ramp compensation)")
            from experiments.utils import append_log
            append_log({
                "experiment_id": pid,
                "phase": 4, "base": "pipeline_19c",
                "description": "join ramp removal — auto-SKIP (join_detector_ok=false)",
                "components": ["ramp_remover"],
                "oof_f1_mean": None, "oof_f1_01": None, "oof_f1_03": None, "oof_f1_05": None,
                "oof_tp": None, "oof_fp": None, "oof_fn": None,
                "delta_oof_vs_base": None, "p_delta_gt0": None,
                "f1_mean": None, "f1_01": None, "f1_03": None, "f1_05": None,
                "tp": None, "fp": None, "fn": None, "false_alarms_per_hour": None,
                "delta_f1": None, "probe_auc": None, "probe_macro_f1": None,
                "gate": False,
                "decision": "SKIPPED_GATE",
                "notes": "join_detector_ok=False → prerequisite not met",
            })
            continue

        # Gate check
        if pid in gated:
            print(f"\n  Checking gate for {pid}…")
            if not check_gate(pid):
                print(f"  {pid}: SKIPPED_GATE")
                from experiments.utils import append_log
                append_log({
                    "experiment_id": pid,
                    "phase": 4, "base": "auto",
                    "description": f"{pid} — gate closed",
                    "components": [],
                    "oof_f1_mean": None, "oof_f1_01": None, "oof_f1_03": None, "oof_f1_05": None,
                    "oof_tp": None, "oof_fp": None, "oof_fn": None,
                    "delta_oof_vs_base": None, "p_delta_gt0": None,
                    "f1_mean": None, "f1_01": None, "f1_03": None, "f1_05": None,
                    "tp": None, "fp": None, "fn": None, "false_alarms_per_hour": None,
                    "delta_f1": None, "probe_auc": None, "probe_macro_f1": None,
                    "gate": False,
                    "decision": "SKIPPED_GATE",
                    "notes": "Gate closed (from gates.json)",
                })
                continue

        decision = run_pipeline(pid, dry_run=args.dry_run)

        if args.dry_run:
            continue

        if decision in ("DROP", "STOPPED"):
            drop_count += 1
            if drop_count >= MAX_CONSEC_DROP:
                print(f"\n🛑 {MAX_CONSEC_DROP} consecutive DROP/STOPPED — halting GĐ4 per stop condition")
                break
        elif decision == "KEEP":
            drop_count = 0  # reset consecutive drop counter

    # Print summary
    print("\n" + "=" * 70)
    print("GĐ4 Pipeline Summary")
    print("=" * 70)
    if LOG_FILE.exists():
        log = json.loads(LOG_FILE.read_text(encoding="utf-8"))
        gd4 = [e for e in log if e.get("phase") == 4]
        print(f"\n{'Pipeline':<15} {'Decision':<15} {'OOF F1':>10} {'Val F1':>10}  Notes")
        print("-" * 75)
        for e in sorted(gd4, key=lambda x: x["experiment_id"]):
            oof = f"{e.get('oof_f1_mean', '-')}"
            val = f"{e.get('f1_mean', '-')}"
            print(f"{e['experiment_id']:<15} {e.get('decision','?'):<15} {oof:>10} {val:>10}  "
                  f"{e.get('notes','')[:40]}")

    # Find best KEEP pipeline
    if LOG_FILE.exists():
        log = json.loads(LOG_FILE.read_text(encoding="utf-8"))
        gd4_keep = [e for e in log if e.get("phase") == 4 and e.get("decision") == "KEEP"]
        if gd4_keep:
            best = max(gd4_keep, key=lambda x: x.get("oof_f1_mean") or 0)
            print(f"\n🏆 Best KEEP pipeline: {best['experiment_id']}")
            print(f"   OOF F1_mean={best.get('oof_f1_mean')}")
            print(f"   Val F1_mean={best.get('f1_mean')}")
            sub = ROOT / f"submission_{best['experiment_id']}.csv"
            if sub.exists():
                print(f"   Submission: {sub}")
            else:
                print(f"   ⚠️  Submission file not found: {sub}")
        else:
            print("\n  No KEEP pipelines — keeping submission_pipeline_19.csv")


if __name__ == "__main__":
    main()
