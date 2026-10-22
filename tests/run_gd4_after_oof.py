"""Script to run after pipeline_19c OOF completes.

Usage (from project root):
    python run_gd4_after_oof.py

This script:
1. Verifies oof_pipeline_19c.csv exists and log has OOF metrics
2. Runs error_audit.py to compute gates.json
3. Runs all GĐ4 pipelines (21 → 25) in sequence via run_gd4_pipelines.py
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

OOF_CSV  = ROOT / "oof_pipeline_19c.csv"
LOG_FILE = ROOT / "experiments_log.json"
REPORTS  = ROOT / "experiments" / "audit" / "reports"


def check_oof_ready() -> bool:
    if not OOF_CSV.exists():
        print(f"❌ OOF CSV not found: {OOF_CSV}")
        return False
    log = json.loads(LOG_FILE.read_text(encoding="utf-8")) if LOG_FILE.exists() else []
    p19c = next((e for e in log if e["experiment_id"] == "pipeline_19c"), None)
    if not p19c:
        print("❌ pipeline_19c not in log")
        return False
    R = p19c.get("oof_f1_mean")
    if R is None:
        print("⚠️  pipeline_19c OOF metrics are null in log")
        # OOF CSV exists but maybe hasn't updated the log yet — still runnable
    print(f"✓ OOF ready. R = {R}, Val = {p19c.get('f1_mean')}")
    return True


def run_step(cmd, description):
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"⚠️  Exit code {result.returncode} — continuing")
    return result.returncode == 0


def main():
    print("=" * 70)
    print("GĐ4 Post-OOF Runner: error_audit + pipelines 21-25")
    print("=" * 70)

    if not check_oof_ready():
        print("\nWait for pipeline_19c OOF to finish, then re-run this script.")
        sys.exit(1)

    # Step 1: Error audit → gates.json
    gates_f = REPORTS / "gates.json"
    if gates_f.exists():
        print(f"\n✓ gates.json already exists — skip audit")
    else:
        run_step(
            [sys.executable,
             str(ROOT / "experiments" / "audit" / "error_audit.py"),
             "--oof-csv", str(OOF_CSV)],
            "Error audit (Step 0.6) → gates.json",
        )

    # Step 2: Verify gates
    if gates_f.exists():
        gates = json.loads(gates_f.read_text(encoding="utf-8"))
        print(f"\nGates: p22={gates.get('gate_p22')}, "
              f"p26={gates.get('gate_p26')}, p27={gates.get('gate_p27')}")
    else:
        print("⚠️  gates.json still missing — pipelines will use safe defaults")

    # Step 3: Run all GĐ4 pipelines
    run_step(
        [sys.executable,
         str(ROOT / "experiments" / "run_gd4_pipelines.py"),
         "--from", "21", "--to", "25"],
        "GĐ4 Pipelines 21–25",
    )

    # Step 4: Check for gate_p27 and run pipeline_27 if open
    if gates_f.exists():
        gates = json.loads(gates_f.read_text(encoding="utf-8"))
        if gates.get("gate_p27"):
            print("\n→ gate_p27 open — running pipeline_27")
            run_step(
                [sys.executable,
                 str(ROOT / "experiments" / "run_gd4_pipelines.py"),
                 "--from", "27", "--to", "27"],
                "GĐ4 Pipeline 27 (FN recovery via g_max proposals)",
            )

    print("\n✓ GĐ4 run complete — check experiments_log.json for results")


if __name__ == "__main__":
    main()
