"""Convenience wrapper for Authorship Verification CLI and Web UI."""
from __future__ import annotations

import sys

from nlu_authorship.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
