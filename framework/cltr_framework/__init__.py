"""Primary package for the CLTR framework workspace."""

import os
from pathlib import Path

_WORK_ROOT = Path(__file__).resolve().parents[1]
(_WORK_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
(_WORK_ROOT / ".cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_WORK_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(_WORK_ROOT / ".cache"))

from .config import CLTRConfig, default_config
from .pipeline import CLTRPipeline

__all__ = ["CLTRConfig", "CLTRPipeline", "default_config"]
