---
name: pdf-pymupdf4llm
description: Local PDF parsing (pymupdf4llm) + LLM structural repair for double-column papers
---

## Parsing

No network required. Parses locally and writes `full.md` to the output directory.

```python
from sub_skills.pymupdf4llm import FallbackParser
from pathlib import Path

result = FallbackParser().parse(Path("<pdf_path>"), Path("<out_dir>"))
# result.markdown  — raw Markdown
# result.images    — extracted image files
# result.out_dir   — output directory (full.md, images/)
```

Double-column academic papers often produce structural artifacts (garbled headings, merged words, scrambled reading order, stray footnotes). Run repair after parsing when these appear.

## Repair

### Configuration

Requires `llm_api_key` and `llm_base_url` in `sub_skills/pymupdf4llm/config.json`. Any service compatible with the Anthropic Messages API is supported.

**If `config.json` is missing**, it is created automatically on first use — the user will be prompted:

```
[pymupdf4llm] config.json not found. Please enter the following:

LLM configuration (any Anthropic Messages API-compatible service):
  LLM API Key: <hidden input>
  LLM Base URL (e.g. https://api.minimaxi.com/anthropic):
  LLM model name, e.g. MiniMax-M2.7 / claude-3-5-haiku-20241022 (optional):
```

To set up manually, create `sub_skills/pymupdf4llm/config.json`:

```json
{
  "llm_api_key": "<your LLM API key>",
  "llm_base_url": "<Anthropic-compatible endpoint>",
  "llm_model": "<model name>"
}
```

Environment variables override config file values:

| Env var | Config key |
|---------|-----------|
| `LLM_API_KEY` | `llm_api_key` |
| `LLM_BASE_URL` | `llm_base_url` |
| `LLM_MODEL` | `llm_model` |
| `ANTHROPIC_API_KEY` | `llm_api_key` |
| `ANTHROPIC_BASE_URL` | `llm_base_url` |

### Usage

```python
from sub_skills.pymupdf4llm import repair, repair_file

# repair a Markdown string
fixed_md = repair(raw_markdown, verbose=True)

# repair a file — writes <stem>_repaired.md alongside the original
out_path = repair_file("output/full.md", verbose=True)
```

### What repair fixes

| Issue | Method |
|-------|--------|
| Spaces injected into headings (`II. T HE`) | LLM rebuilds (`## II. The ...`) |
| Merged words from column breaks (`domainspecific`) | LLM restores (`domain-specific`) |
| Scrambled double-column reading order | LLM reorders |
| Footnotes / author / affiliation lines | LLM removes |
| Caption fragment leftovers | Rule-based pre-processing |
| Soft line breaks within paragraphs | Rule-based pre-processing |
| Isolated formula fragments | LLM wraps in ` ```math ``` ` |
| Sentences split across chunk boundaries | Rule-based seam stitching |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_key` | config.json | LLM API key |
| `base_url` | config.json | LLM base URL |
| `model` | config.json / `claude-3-5-haiku-20241022` | LLM model name |
| `max_chars_per_chunk` | `3000` | Max characters per LLM call |
| `seam_context` | `2` | Paragraphs inspected at each chunk boundary |
| `max_workers` | `4` | Concurrent threads |

For design details see `pdf_repair.md`.

## Errors

- Missing API key → raises `ValueError`; configure `llm_api_key` and retry
- Network / LLM error → raises exception with the server message
