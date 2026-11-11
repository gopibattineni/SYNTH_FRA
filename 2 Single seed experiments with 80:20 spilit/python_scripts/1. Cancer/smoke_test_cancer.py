#!/usr/bin/env python3
"""Quick smoke test: bootstrap + data load (notebook cells 0-1 logic)."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "cancer.py"


def main() -> int:
    print("Smoke test: executing cells 0-1 from cancer.py ...")
    source = SCRIPT.read_text(encoding="utf-8")
    bootstrap, rest = source.split("# END RUNTIME BOOTSTRAP", 1)

    # Execute bootstrap
    g: dict = {"__name__": "__smoke__", "__file__": str(SCRIPT)}
    exec(compile(bootstrap, str(SCRIPT), "exec"), g)

    # Extract cells 0 and 1 only
    parts = rest.split("# ========================================================================\n# Cell ")
    chunk = "# ========================================================================\n# Cell ".join(parts[:3])
    exec(compile(chunk, str(SCRIPT), "exec"), g)

    cancer_data = g["cancer_data"]
    print(f"OK: loaded cancer_data shape={cancer_data.shape}")
    print(f"OK: target_col={g['target_col']!r}, SEED={g['SEED']}, N_SAMPLES={g['N_SAMPLES']}")
    print(f"OK: cwd={Path.cwd()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
