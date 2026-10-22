"""Pipeline 17: CutMix/noise-augmented model (same p16 structure, thr=0.10)."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from abc2027.baseline import window_features
from experiments.utils import load_val, run_pipeline
from experiments.models.lightgbm_classifier import load_classifier_augmented
from experiments.pipelines.pipeline_16 import run


def main():
    streams = load_val()
    model = load_classifier_augmented(feat_fn=window_features, label="base+aug")
    rows = run(streams, model, threshold=0.10)
    run_pipeline(17, "p09 + feature-aug training (noise+scale+mixup, 9 features, thr=0.10)", rows, 0.6091)


if __name__ == "__main__":
    main()
