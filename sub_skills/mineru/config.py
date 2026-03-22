"""Load and validate mineru/config.json."""

import json
import os
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "config.json"


def _prompt(label: str, secret: bool = False) -> str:
    if secret:
        try:
            import getpass
            return getpass.getpass(f"  {label}: ").strip()
        except Exception:
            pass
    return input(f"  {label}: ").strip()


def ensure_config() -> None:
    if _CONFIG_PATH.exists():
        return

    print("[mineru] config.json 不存在，请填写以下配置：")
    print()

    mineru_api_key = _prompt("MinerU API Key", secret=True)

    cfg = {
        "mineru_api_key":  mineru_api_key,
        "mineru_base_url": "https://mineru.net/api/v4",
    }

    _CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[mineru] 配置已保存至 {_CONFIG_PATH}\n")


def load_config() -> dict:
    ensure_config()

    cfg: dict = {}
    if _CONFIG_PATH.exists():
        cfg = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))

    env_map = {
        "MINERU_API_KEY":      "mineru_api_key",
        "MINERU_BASE_URL":     "mineru_base_url",
    }
    for env_key, cfg_key in env_map.items():
        val = os.environ.get(env_key)
        if val:
            cfg[cfg_key] = val

    return cfg
