---
name: pdf-pymupdf4llm
description: Local PDF parsing (pymupdf4llm) + LLM structural repair for double-column papers
---

# How To Use

## Configuration

Requires `llm_api_key` and `llm_base_url` in `sub_skills/pymupdf4llm/config.json`. Any service compatible with the Anthropic Messages API is supported.

Please manually create `sub_skills/pymupdf4llm/config.json`:

```json
{
  "llm_api_key": "<your LLM API key>",
  "llm_base_url": "<Anthropic-compatible endpoint>",
  "llm_model": "<model name>"
}
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pdf_path` | required | Path to the PDF file to parse |
| `out_dir` | required | Directory to save the output Markdown |

## Usage

Call **Parsing** first, then call **Repair**.

## Errors

- Analysis the errors. 
- If it is a config error, prompt user to configure interactively.
- For other errors, directly print the error message.


# Tools

## Parsing

Parses locally with pdf-pymupdf4llm.

### Usage

```python
from sub_skills.pymupdf4llm import FallbackParser
from pathlib import Path

FallbackParser().parse(Path("<pdf_path>"), Path("<out_dir>/<pdf_name>.md"))
# <pdf_name> is the original PDF name in <pdf_path> without the .pdf extension
```

## Repair

Repairs the Content with LLM, see `pdf_repair.md` if needed.

### Usage

```python
from sub_skills.pymupdf4llm import repair, repair_file

out_path = repair_file("<out_dir>/<pdf_name>.md", verbose=True)
# Will save to <out_dir>/<pdf_name>_repaired.md
```

### More Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_chars_per_chunk` | `3000` | Max characters per LLM call |
| `seam_context` | `2` | Paragraphs inspected at each chunk boundary |
| `max_workers` | `4` | Concurrent threads |

