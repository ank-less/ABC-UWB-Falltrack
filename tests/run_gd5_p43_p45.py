"""Sequential runner: P43 → P44 → P45.

Runs each pipeline; if val F1 >= 0.80, stops and reports victory.
Each pipeline reads the log to pick up the best base automatically.
"""
import subprocess
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent

def run_pipeline(name: str) -> dict:
    script = ROOT / "experiments" / "pipelines" / f"{name}.py"
    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(script)],
        capture_output=False,
        text=True,
    )
    log_path = ROOT / "experiments_log.json"
    if log_path.exists():
        log = json.loads(log_path.read_text(encoding="utf-8"))
        entry = next((e for e in log if e["experiment_id"] == name), {})
        return entry
    return {}


if __name__ == "__main__":
    pipelines = ["pipeline_43", "pipeline_44", "pipeline_45"]
    for name in pipelines:
        print(f"\n{'='*60}")
        print(f"Starting {name}…")
        print(f"{'='*60}\n")
        entry = run_pipeline(name)
        val_f1 = entry.get("f1_mean")
        print(f"\n→ {name}: decision={entry.get('decision')}  val_F1={val_f1}")
        if val_f1 and val_f1 >= 0.80:
            print(f"\n🎯 TARGET REACHED! Val F1 = {val_f1:.4f} >= 0.80")
            print(f"   Best pipeline: {name}")
            break
    else:
        log = json.loads((ROOT / "experiments_log.json").read_text(encoding="utf-8"))
        best = max((e for e in log if e.get("f1_mean")), key=lambda e: e["f1_mean"])
        print(f"\nCompleted P43-P45. Best val: {best['experiment_id']} F1={best['f1_mean']:.4f}")
        print("Continue with P46+")
