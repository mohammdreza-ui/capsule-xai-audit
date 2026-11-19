"""Library-wide logger."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "capsule_xai_audit") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        root = logging.getLogger("capsule_xai_audit")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name)
