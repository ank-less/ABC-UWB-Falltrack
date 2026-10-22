"""Run P37-P40 sequentially for GD5.

Run after P36 completes. Each pipeline:
  1. Checks if already logged (skip if so)
  2. Runs the pipeline
  3. Stops early if F1 >= 0.80 (goal reached)
  4. Stops early if 3 consecutive DROPs
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

GOAL_F1 = 0.80
PIPELINES = [37, 38, 39, 40]


def get_decision(pid: int) -> str | None:
    log_path = ROOT / "experiments_log.json"
    if not log_path.exists():
        return None
    log = json.loads(log_path.read_text(encoding="utf-8"))
    for entry in log:
        if entry["experiment_id"] == f"pipeline_{pid:02d}":
            return entry.get("decision")
    return None


def get_val_f1(pid: int) -> float | None:
    log_path = ROOT / "experiments_log.json"
    if not log_path.exists():
        return None
    log = json.loads(log_path.read_text(encoding="utf-8"))
    for entry in log:
        if entry["experiment_id"] == f"pipeline_{pid:02d}":
            return entry.get("f1_mean")
    return None


def main():
    drops = 0
    best_val = 0.0

    for pid in PIPELINES:
        script = ROOT / "experiments" / "pipelines" / f"pipeline_{pid:02d}.py"
        if not script.exists():
            print(f"[P{pid}] Script not found: {script}")
            break

        # Check if already done
        existing = get_decision(pid)
        if existing is not None:
            print(f"[P{pid}] Already logged: decision={existing}")
            val_f1 = get_val_f1(pid)
            if val_f1 and val_f1 > best_val:
                best_val = val_f1
            if existing == "DROP":
                drops += 1
            else:
                drops = 0
            if best_val >= GOAL_F1:
                print(f"[P{pid}] GOAL REACHED: val F1={best_val:.4f} >= {GOAL_F1}")
                break
            continue

        # Run pipeline
        print(f"\n{'='*70}")
        print(f"Running pipeline_{pid:02d}...")
        print(f"{'='*70}")
        result = subprocess.run(
            [sys.executable, str(script)],
            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
        )
        if result.returncode != 0:
            print(f"[P{pid}] FAILED with exit code {result.returncode}")
            drops += 1
        else:
            dec = get_decision(pid)
            val_f1 = get_val_f1(pid)
            print(f"[P{pid}] decision={dec}  val_f1={val_f1}")
            if val_f1 and val_f1 > best_val:
                best_val = val_f1
            if dec == "DROP":
                drops += 1
            else:
                drops = 0

        if best_val >= GOAL_F1:
            print(f"\nGOAL REACHED: val F1={best_val:.4f} >= {GOAL_F1}")
            break
        if drops >= 3:
            print(f"\nEarly stop: 3 consecutive DROPs")
            break

    print(f"\nFinal best val F1 = {best_val:.4f}")


if __name__ == "__main__":
    main()
