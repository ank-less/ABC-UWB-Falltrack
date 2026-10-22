"""Build the stream manifest and the segment table (no CIR values) from the dataset package."""

from __future__ import annotations

import argparse
import sys

import _bootstrap
from abc_uwb_falltrack.config import get, load_config
from abc_uwb_falltrack.experiment import run_manifest, setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _bootstrap.add_common_arguments(parser)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.override, args.smoke)
    setup_logging()
    manifest, segments = run_manifest(cfg)
    print(f"manifest: {len(manifest)} streams -> {get(cfg, 'data.manifest_path')}")
    print(f"segments: {len(segments)} rows, {segments['subject_group'].nunique()} subject groups -> {get(cfg, 'data.segments_path')}")
    print(manifest.groupby("official_split")[["n_events", "duration_s"]].sum().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
