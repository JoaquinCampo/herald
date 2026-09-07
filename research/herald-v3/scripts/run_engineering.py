#!/usr/bin/env python3
"""Launch the bounded HERALD v3 engineering acceptance runner."""

# The import intentionally follows the local source-path bootstrap.
# ruff: noqa: E402, I001

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from herald_v3.engineering.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
