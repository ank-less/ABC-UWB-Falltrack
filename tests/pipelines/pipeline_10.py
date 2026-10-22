"""Pipeline 10: 3 radars independent p04 + majority-vote fusion (2/3)."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from abc2027.baseline import frame_energy
from experiments.utils import load_val, make_rows, run_pipeline, energy_events_raw
from experiments.preprocessing.chebyshev_filter import chebyshev_bp
from experiments.postprocessing.temporal_smoothing import smooth
from experiments.postprocessing.wbf_ensemble import late_fusion_majority


def _p04_single_radar(radar_cir):
    """Run p04 on a single-radar CIR (1, T, 368)."""
    filtered = chebyshev_bp(radar_cir)
    energy = smooth(frame_energy(filtered), sigma_s=0.3)
    return energy_events_raw(energy, pct=98.0, min_dur=0.6, merge_gap=0.8)


def run(streams):
    rows = []
    for sid, cir in streams.items():
        n_radars = cir.shape[0]
        events_r = [_p04_single_radar(cir[r:r+1, :, :]) for r in range(n_radars)]
        events = late_fusion_majority(events_r, min_support=2)
        rows.extend(make_rows(sid, events))
    return rows


def main():
    streams = load_val()
    rows = run(streams)
    run_pipeline(10, "3_radar_independent_p04 + majority_vote_fusion(2/3)", rows, 0.5230)


if __name__ == "__main__":
    main()
