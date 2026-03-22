"""Load and validate mineru/config.json."""

import json
import os
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config() -> dict:
    """Load config from file, then override with environment variables.

    Raises:
        FileNotFoundError: if config.json is missing and MINERU_API_KEY env var is not set.
        KeyError: if mineru_api_key is missing from config.
    """
    cfg: dict = {}

    if _CONFIG_PATH.exists():
        cfg = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))

    env_map = {
        "MINERU_API_KEY":  "mineru_api_key",
        "MINERU_BASE_URL": "mineru_base_url",
    }
    for env_key, cfg_key in env_map.items():
        val = os.environ.get(env_key)
        if val:
            cfg[cfg_key] = val

    if not cfg:
        raise FileNotFoundError(
            f"[mineru] config.json not found at {_CONFIG_PATH}. "
            "Create it with mineru_api_key, or set the MINERU_API_KEY environment variable."
        )
    if not cfg.get("mineru_api_key"):
        raise KeyError(
            "[mineru] mineru_api_key is missing. "
            f"Add it to {_CONFIG_PATH} or set the MINERU_API_KEY environment variable."
        )

    return cfg
