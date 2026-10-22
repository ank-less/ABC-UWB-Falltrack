"""Pipeline 12: Chebyshev + 3-state HSMM (fall mask + forward lying extension 2s)."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from abc2027.baseline import frame_energy
from experiments.utils import load_val, make_rows, run_pipeline
from experiments.preprocessing.chebyshev_filter import chebyshev_bp
from experiments.postprocessing.temporal_smoothing import smooth
from experiments.models.hsmm_detector import hsmm_events


def run(streams):
    rows = []
    for sid, cir in streams.items():
        filtered = chebyshev_bp(cir)
        energy = smooth(frame_energy(filtered), sigma_s=0.3)
        events = hsmm_events(energy)
        rows.extend(make_rows(sid, events))
    return rows


def main():
    streams = load_val()
    rows = run(streams)
    run_pipeline(12, "Chebyshev + 3state_HSMM: fall_mask + forward_lying_dilation(2s)", rows, 0.4640)


if __name__ == "__main__":
    main()
