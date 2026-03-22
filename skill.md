---
name: pdf
description: Parse a PDF into Markdown for Agent/LLM consumption
---

Parse a user-specified PDF into structured Markdown.

## Sub-skills

- **MinerU** — cloud-based, high-quality parsing with formulas, tables, and images → `sub_skills/mineru/skill.md`
- **pymupdf4llm + repair** — local parsing with LLM-based structural repair for double-column papers → `sub_skills/pymupdf4llm/skill.md`

## Parsing Strategy

1. Try **MinerU** first. If `mineru_api_key` is not configured or the API call fails, fall back to **pymupdf4llm**.
2. After pymupdf4llm parsing, inspect the output. If the paper is double-column and shows structural artifacts (garbled headings, merged words, scrambled order, stray footnotes), run **repair**.

## Configuration Setup

Each sub_skill manages its own `config.json`. On first use, missing config is detected automatically and the user is prompted interactively. To set up manually:

**MinerU** — create `sub_skills/mineru/config.json`:
```json
{
  "mineru_api_key": "<your MinerU API key>",
  "mineru_base_url": "https://mineru.net/api/v4"
}
```

**pymupdf4llm repair** — create `sub_skills/pymupdf4llm/config.json`:
```json
{
  "llm_api_key": "<your LLM API key>",
  "llm_base_url": "<Anthropic-compatible endpoint>",
  "llm_model": "<model name>"
}
```
