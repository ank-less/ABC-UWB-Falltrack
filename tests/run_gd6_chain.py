"""GD6 pipeline chain runner: p49 → p51 → p50 in sequence.

Runs each pipeline, checks KEEP/DROP, updates CLAUDE.md summary, and chains.
Designed to run after p48 completes.

Usage:
    python experiments/run_gd6_chain.py [--start p49]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

PIPELINES = ["pipeline_49", "pipeline_51", "pipeline_50"]
BASE_OOF  = 0.7423   # p46 base

def read_log_entry(eid: str) -> dict:
    f = ROOT / "experiments_log.json"
    if not f.exists():
        return {}
    data = json.loads(f.read_text(encoding="utf-8"))
    return next((e for e in data if e.get("experiment_id") == eid), {})


def run_pipeline(mod: str) -> dict:
    print(f"\n{'='*70}")
    print(f"Starting {mod}...")
    print(f"{'='*70}\n")
    result = subprocess.run(
        [sys.executable, "-m", f"experiments.pipelines.{mod}"],
        cwd=str(ROOT),
        env={**__import__("os").environ, "PYTHONUTF8": "1"},
    )
    if result.returncode != 0:
        print(f"ERROR: {mod} exited with code {result.returncode}")
        return {"error": True, "returncode": result.returncode}
    entry = read_log_entry(mod.replace("_", "_", 1))
    if not entry:
        # Try variant naming
        for variant in (mod, mod.replace("pipeline_", "pipeline_")):
            entry = read_log_entry(variant)
            if entry:
                break
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="pipeline_49",
                        help="First pipeline to run (default: pipeline_49)")
    args = parser.parse_args()

    start_idx = 0
    for i, p in enumerate(PIPELINES):
        if p == args.start:
            start_idx = i
            break

    # Dynamically find current best KEEP pipeline from log
    best_base_oof = BASE_OOF
    best_base_id  = "pipeline_46"
    f = ROOT / "experiments_log.json"
    if f.exists():
        log = json.loads(f.read_text(encoding="utf-8"))
        gd6_keep = [e for e in log
                    if e.get("phase") in (5, 6) and e.get("decision") == "KEEP"
                    and isinstance(e.get("oof_f1_mean"), float)
                    and e.get("oof_f1_mean", 0) > BASE_OOF]
        if gd6_keep:
            best_entry = max(gd6_keep, key=lambda e: e.get("oof_f1_mean", 0))
            best_base_oof = best_entry["oof_f1_mean"]
            best_base_id  = best_entry["experiment_id"]
            print(f"Auto-detected best base from log: {best_base_id}  OOF={best_base_oof:.4f}")
    consecutive_drops = 0

    print(f"\nGD6 Chain: {' -> '.join(PIPELINES[start_idx:])}")
    print(f"Base: {best_base_id}  OOF={best_base_oof:.4f}")
    print(f"KEEP criterion: OOF F1_mean >= {best_base_oof + 0.02:.4f}\n")

    results_summary = []
    for pid in PIPELINES[start_idx:]:
        entry = run_pipeline(pid)
        if entry.get("error"):
            print(f"{pid}: CRASHED (exit {entry.get('returncode')})")
            consecutive_drops += 1
            results_summary.append({"id": pid, "result": "CRASHED"})
        else:
            oof = entry.get("oof_f1_mean", 0.0)
            # Re-evaluate KEEP/DROP based on CURRENT best base, not pipeline's own criterion
            # (individual pipelines compare against p46; chain uses dynamic best)
            chain_keep = oof >= best_base_oof + 0.02
            decision = "KEEP" if chain_keep else "DROP"
            delta = round(oof - best_base_oof, 4)
            print(f"\n{pid}: OOF={oof:.4f}  delta={delta:+.4f}  chain_decision={decision}  (logged_decision={entry.get('decision')})")

            if decision == "KEEP":
                best_base_oof = oof
                best_base_id  = pid
                consecutive_drops = 0
                print(f"  New best: {best_base_id}  OOF={best_base_oof:.4f}")
                if oof >= 0.80:
                    print(f"\n*** TARGET REACHED: OOF F1_mean={oof:.4f} >= 0.80 ***")
                    results_summary.append({"id": pid, "oof": oof, "result": "TARGET_REACHED"})
                    break
            else:
                consecutive_drops += 1
                results_summary.append({"id": pid, "oof": oof, "result": decision})

            if consecutive_drops >= 3:
                print(f"\nStopping: {consecutive_drops} consecutive DROPs")
                break

    print("\n" + "=" * 70)
    print("GD6 Chain Summary:")
    for r in results_summary:
        oof_str = f"  OOF={r['oof']:.4f}" if "oof" in r else ""
        print(f"  {r['id']}: {r['result']}{oof_str}")
    print(f"Best base: {best_base_id}  OOF={best_base_oof:.4f}")
    if best_base_oof >= 0.80:
        print("TARGET REACHED: OOF F1_mean >= 0.80")
    else:
        print(f"Still {0.80 - best_base_oof:.4f} below target (0.80)")


if __name__ == "__main__":
    main()
