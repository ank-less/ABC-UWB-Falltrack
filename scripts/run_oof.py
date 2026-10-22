#!/usr/bin/env python3
"""Run train-only grouped OOF tuning for the quantum-inspired readout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abc_uwb_falltrack.oof import load_config, run_oof  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "quantum_oof.yaml",
        help="Experiment configuration (default: configs/quantum_oof.yaml)",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    result = run_oof(load_config(config_path), config_path)
    target = float(result["requested_internal_target_mean_event_f1"])
    achieved = float(result["best_oof"]["ranking_score"]["f1_mean"])
    if achieved < target:
        print(f"internal_target_not_reached={target:.4f}")
    else:
        print(f"internal_oof_target_reached={target:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
