"""Load and validate pymupdf4llm/config.json."""

import json
import os
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config() -> dict:
    """Load config from file, then override with environment variables.

    Raises:
        FileNotFoundError: if config.json is missing and no env vars are set.
        KeyError: if llm_api_key or llm_base_url is missing.
    """
    cfg: dict = {}

    if _CONFIG_PATH.exists():
        cfg = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))

    env_map = {
        "LLM_API_KEY":        "llm_api_key",
        "LLM_BASE_URL":       "llm_base_url",
        "LLM_MODEL":          "llm_model",
        "ANTHROPIC_API_KEY":  "llm_api_key",
        "ANTHROPIC_BASE_URL": "llm_base_url",
    }
    for env_key, cfg_key in env_map.items():
        val = os.environ.get(env_key)
        if val:
            cfg[cfg_key] = val

    if not cfg:
        raise FileNotFoundError(
            f"[pymupdf4llm] config.json not found at {_CONFIG_PATH}. "
            "Create it with llm_api_key and llm_base_url, or set the LLM_API_KEY / LLM_BASE_URL environment variables."
        )
    for key in ("llm_api_key", "llm_base_url", "llm_model"):
        if not cfg.get(key):
            raise KeyError(
                f"[pymupdf4llm] {key} is missing. "
                f"Add it to {_CONFIG_PATH} or set the corresponding environment variable (LLM_API_KEY / LLM_BASE_URL / LLM_MODEL)."
            )

    return cfg
