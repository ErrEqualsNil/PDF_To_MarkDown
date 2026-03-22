"""Load and validate pymupdf4llm/config.json."""

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

    print("[pymupdf4llm] config.json 不存在，请填写以下配置（直接回车跳过可选项）：")
    print()
    print("LLM 配置（支持任何兼容 Anthropic Messages API 的服务）：")

    llm_api_key  = _prompt("LLM API Key", secret=True)
    llm_base_url = _prompt("LLM Base URL，如 https://api.minimaxi.com/anthropic")
    llm_model    = _prompt("LLM 模型名称，如 MiniMax-M2.7 / claude-3-5-haiku-20241022（可选）")

    cfg = {
        "llm_api_key":  llm_api_key,
        "llm_base_url": llm_base_url,
    }
    if llm_model:
        cfg["llm_model"] = llm_model

    _CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[pymupdf4llm] 配置已保存至 {_CONFIG_PATH}\n")


def load_config() -> dict:
    ensure_config()

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

    return cfg
