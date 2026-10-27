"""The setup folder a run uses when a tool is not given one.

agent.config.load_config exports the run's `inputs.setup` as DEXAGENT_SETUP, so a tool called by any stage (in process
or in another interpreter) defaults to the run's bench; without it, the default bench in setup/.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_SETUP = Path(__file__).resolve().parents[2] / "setup"


def setup_dir() -> Path:
    return Path(os.environ.get("DEXAGENT_SETUP") or DEFAULT_SETUP)
