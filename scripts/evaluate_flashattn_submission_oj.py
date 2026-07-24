#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_flashattn_submission import main as evaluate_main


def main() -> None:
    # OJ-specific wrapper: keep the implementation shared, but make the
    # official contest profile the default and harder to forget.
    if "--task-profile" not in sys.argv:
        sys.argv.extend(["--task-profile", "contest"])
    evaluate_main()


if __name__ == "__main__":
    main()
