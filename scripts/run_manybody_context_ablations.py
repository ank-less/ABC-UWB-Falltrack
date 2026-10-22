#!/usr/bin/env python3
"""Hai ablation readout đã định trước, cùng R32 source-context folds.

Không thay lựa chọn mô hình chính; giữ cùng QFI decoder, hyperparameters,
duration prior và seed của từng outer fold. Không đọc validation/test.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

import run_manybody_ablations as ablation
import run_manybody_context_nested as context


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="configs/manybody_source_context_nested_r32.yaml")
    args, _ = parser.parse_known_args()
    config = yaml.safe_load((context.ROOT / args.config).read_text())
    context.validate_context_provenance(config)
    original_hashes = ablation.code_hashes

    def contextual_code_hashes():
        values = original_hashes()
        for name in ("scripts/run_manybody_context_nested.py", "scripts/build_source_disjoint_context_package.py",
                     "scripts/run_manybody_context_ablations.py"):
            values[name] = ablation.digest(context.ROOT / name)
        return values

    ablation.code_hashes = contextual_code_hashes
    ablation.signature = context.context_signature
    ablation.main()


if __name__ == "__main__":
    main()
