---
name: pdf-mineru
description: Parse a PDF via the MinerU API into structured Markdown (formulas, tables, images)
---

## Configuration

Requires `mineru_api_key` in `sub_skills/mineru/config.json` (or env var `MINERU_API_KEY`).

**If `config.json` is missing**, it is created automatically on first use — the user will be prompted:

```
[mineru] config.json not found. Please enter the following:

  MinerU API Key: <hidden input>
```

To set up manually, create `sub_skills/mineru/config.json`:

```json
{
  "mineru_api_key": "<your MinerU API key>",
  "mineru_base_url": "https://mineru.net/api/v4"
}
```

Environment variables override config file values:

| Env var | Config key |
|---------|-----------|
| `MINERU_API_KEY` | `mineru_api_key` |
| `MINERU_BASE_URL` | `mineru_base_url` |

## Usage

```python
from sub_skills.mineru import MinerUParser
from sub_skills.mineru.config import load_config
from pathlib import Path

cfg = load_config()
result = MinerUParser(cfg).parse(Path("<pdf_path>"), Path("<out_dir>"))
# result.markdown  — full Markdown (formulas in LaTeX, tables in HTML)
# result.images    — [Path, ...]  extracted image files
# result.pages     — page count
# result.out_dir   — output directory (full.md, images/, result.zip)
```

## API Flow

1. `POST /file-urls/batch` — obtain a signed upload URL
2. `PUT <signed_url>` — upload the PDF
3. `POST /extract/task/batch` — submit extraction task (formulas, tables, OCR enabled)
4. `GET /extract-results/batch/{id}` — poll every 5 s, up to 10 min
5. Download `full_zip_url` → unpack to output directory

## Timing

30–120 seconds depending on page count.

## Errors

- Missing API key → raises `KeyError`; configure `mineru_api_key` and retry
- Network timeout / extraction failure → raises `RuntimeError` with the server message
