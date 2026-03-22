"""
MinerU Precise API client.

Flow:
  1. POST /file-urls/batch          → batch_id + signed PUT URL
  2. PUT  {signed_url}              → upload file
  3. POST /extract/task/batch       → submit extraction task
  4. GET  /extract-results/batch/{} → poll until done
  5. Download full_zip_url          → unpack ZIP to out_dir
"""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

import requests

from .models import ParseResult

_POLL_INTERVAL = 5    # seconds
_POLL_TIMEOUT  = 600  # seconds


class MinerUParser:
    def __init__(self, cfg: dict):
        self.base    = cfg.get("mineru_base_url", "https://mineru.net/api/v4")
        self.headers = {
            "Authorization": f"Bearer {cfg['mineru_api_key']}",
            "Content-Type":  "application/json",
        }

    # ── public ────────────────────────────────────────────────────────────────

    def parse(self, pdf_path: Path, out_dir: Path) -> ParseResult:
        batch_id, signed_url = self._get_upload_url(pdf_path.name)
        self._upload(pdf_path, signed_url)
        self._submit(batch_id, signed_url)
        item = self._poll(batch_id)
        return self._download_and_build(item, out_dir)

    # ── private ───────────────────────────────────────────────────────────────

    def _get_upload_url(self, filename: str) -> tuple[str, str]:
        r = requests.post(
            f"{self.base}/file-urls/batch",
            headers=self.headers,
            json={"files": [{"name": filename, "data_id": "pdf_skill"}]},
            timeout=30,
        )
        r.raise_for_status()
        body = r.json()
        if body["code"] != 0:
            raise RuntimeError(f"file-urls/batch error: {body}")
        return body["data"]["batch_id"], body["data"]["file_urls"][0]

    def _upload(self, pdf_path: Path, signed_url: str):
        with open(pdf_path, "rb") as f:
            r = requests.put(signed_url, data=f, timeout=120)
        if r.status_code not in (200, 204):
            raise RuntimeError(f"Upload failed {r.status_code}: {r.text[:200]}")

    def _submit(self, batch_id: str, signed_url: str):
        file_url = signed_url.split("?")[0]
        r = requests.post(
            f"{self.base}/extract/task/batch",
            headers=self.headers,
            json={
                "batch_id":       batch_id,
                "enable_formula": True,
                "enable_table":   True,
                "language":       "en",
                "files": [{"url": file_url, "data_id": "pdf_skill", "is_ocr": True}],
            },
            timeout=30,
        )
        r.raise_for_status()
        body = r.json()
        if body["code"] != 0:
            raise RuntimeError(f"extract/task/batch error: {body}")

    def _poll(self, batch_id: str) -> dict:
        for _ in range(_POLL_TIMEOUT // _POLL_INTERVAL):
            time.sleep(_POLL_INTERVAL)
            r = requests.get(
                f"{self.base}/extract-results/batch/{batch_id}",
                headers=self.headers,
                timeout=30,
            )
            r.raise_for_status()
            body = r.json()
            if body["code"] != 0:
                raise RuntimeError(f"poll error: {body}")
            results = body["data"].get("extract_result", [])
            if not results:
                continue
            item  = results[0]
            state = item.get("state", "unknown")
            if state == "done":
                return item
            if state == "failed":
                raise RuntimeError(f"Extraction failed: {item.get('err_msg')}")
        raise TimeoutError("MinerU timed out")

    def _download_and_build(self, item: dict, out_dir: Path) -> ParseResult:
        zip_url = item["full_zip_url"]
        r = requests.get(zip_url, timeout=120)
        r.raise_for_status()

        zip_path = out_dir / "result.zip"
        zip_path.write_bytes(r.content)

        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir)

        # Locate outputs
        md_files  = list(out_dir.rglob("*.md"))
        img_files = [f for ext in ("*.jpg", "*.png", "*.jpeg")
                     for f in out_dir.rglob(ext)]

        markdown = md_files[0].read_text(encoding="utf-8") if md_files else ""
        return ParseResult(markdown=markdown, images=img_files, out_dir=out_dir)
