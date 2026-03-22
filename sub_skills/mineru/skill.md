---
name: pdf-mineru
description: Parse a PDF via the MinerU API into structured Markdown (formulas, tables, images)
---

# How To Use

## Configuration

Requires `mineru_api_key` in `sub_skills/mineru/config.json`.

To set up manually, create `sub_skills/mineru/config.json`:

```json
{
  "mineru_api_key": "<your MinerU API key>",
  "mineru_base_url": "https://mineru.net/api/v4"
}
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pdf_path` | required | Path to the PDF file to parse |
| `out_dir` | required | Directory to save the output, same as PDF name |
| `timeout` | 120 seconds | Maximum time to wait for the API response |


## Usage

```python
from sub_skills.mineru import MinerUParser
from sub_skills.mineru.config import load_config
from pathlib import Path

cfg = load_config()
result = MinerUParser(cfg).parse(Path("<pdf_path>"), Path("<out_dir>"))
```

## Latency

30–120 seconds depending on page count.

## Errors

- Analyze the errors.
- If it is a config error, prompt user to configure interactively.
- For other errors, directly print the error message.
