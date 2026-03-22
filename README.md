# pdf-skill

Parse academic PDFs into clean Markdown, designed for Agent / LLM pipelines.

[中文文档](README_zh.md)

## Features

- **MinerU API** — precise parsing with formulas (LaTeX), tables (HTML), and images
- **pymupdf4llm** — local parsing, no network required
- **LLM repair** — four-stage pipeline that fixes double-column artifacts: garbled headings, merged words, scrambled order, footnotes, and caption fragments

## Installation

```bash
pip install -e .
```

Dependencies: `pymupdf4llm`, `pymupdf`, `requests`, `anthropic`

## Configuration

Each sub_skill has its own config file. On first run, you will be prompted to fill in the values. You can also edit the files directly:

**`sub_skills/mineru/config.json`**
```json
{
  "mineru_api_key": "...",
  "mineru_base_url": "https://mineru.net/api/v4"
}
```

**`sub_skills/pymupdf4llm/config.json`**
```json
{
  "llm_api_key": "...",
  "llm_base_url": "https://api.minimaxi.com/anthropic",
  "llm_model": "MiniMax-M2.7"
}
```

Any LLM service compatible with the Anthropic Messages API is supported. Environment variables (`MINERU_API_KEY`, `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`) override config file values.

## Usage

### Parse with MinerU

```python
from sub_skills.mineru import MinerUParser
from sub_skills.mineru.config import load_config
from pathlib import Path

cfg = load_config()
result = MinerUParser(cfg).parse(Path("paper.pdf"), Path("output/"))
print(result.markdown)  # full Markdown
print(result.images)    # [Path, ...]
```

### Parse locally with pymupdf4llm

```python
from sub_skills.pymupdf4llm import FallbackParser
from pathlib import Path

result = FallbackParser().parse(Path("paper.pdf"), Path("output/"))
print(result.markdown)
```

### Repair double-column Markdown

```python
from sub_skills.pymupdf4llm import repair, repair_file

# repair a string
fixed_md = repair(raw_markdown, verbose=True)

# repair a file (writes <stem>_repaired.md alongside the original)
out_path = repair_file("output/full.md", verbose=True)
```

Repair parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | config.json / `claude-3-5-haiku-20241022` | LLM model |
| `max_chars_per_chunk` | `3000` | Max characters per LLM call |
| `seam_context` | `2` | Paragraphs to inspect at chunk boundaries |
| `max_workers` | `4` | Concurrent threads |

## Example

The [`example/`](example/) folder contains a sample run on the Kimi *Attention Residuals* paper ([arXiv 2603.15031](https://arxiv.org/abs/2603.15031)), a 21-page double-column technical report.

```
example/
  example.pdf               # source PDF
  example_mineru/
    example.md              # parsed by MinerU API
  example_pymupdf/
    example.md              # parsed by pymupdf4llm (before repair)
```

**Run it yourself:**

```python
from sub_skills.mineru import MinerUParser
from sub_skills.mineru.config import load_config
from sub_skills.pymupdf4llm import FallbackParser
from pathlib import Path

# MinerU
result = MinerUParser(load_config()).parse(Path("example/example.pdf"), Path("example/example_mineru"))

# pymupdf4llm
result = FallbackParser().parse(Path("example/example.pdf"), Path("example/example_pymupdf"))
```

**Output comparison (first heading):**

| | MinerU | pymupdf4llm |
|---|---|---|
| Title | `# TECHNICAL REPORT OF ATTENTION RESIDUALS` | `## ATTENTION RESIDUALS` |
| Formula | `$h_l = \sum_i \alpha_{i \to l} v_i$` (LaTeX) | inline text fragments |
| Images | not saved | not saved |

## Project Structure

```
sub_skills/
  mineru/
    config.json      # MinerU API key
    config.py        # config loader
    models.py        # ParseResult dataclass
    mineru.py        # MinerU API client
    skill.md         # usage reference
  pymupdf4llm/
    config.json      # LLM config
    config.py        # config loader
    models.py        # ParseResult dataclass
    fallback.py      # local PDF parser
    repair.py        # LLM repair pipeline
    pdf_repair.md    # repair design document
    skill.md         # usage reference
skill.md             # top-level orchestration (natural language)
```
