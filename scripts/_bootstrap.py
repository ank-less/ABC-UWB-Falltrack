"""Make ``src/`` importable when the package is not installed with ``pip install -e .``."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    try:
        import abc_uwb_falltrack  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(SRC))


def add_common_arguments(parser) -> None:
    parser.add_argument("--config", default="configs/baseline.yaml", help="YAML config (default: %(default)s)")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override a config value, e.g. --override experiment.max_streams_per_split=1 (repeatable)",
    )
    parser.add_argument("--smoke", action="store_true", help="apply experiment.smoke limits (few streams, short)")
