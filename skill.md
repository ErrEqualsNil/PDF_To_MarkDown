---
name: pdf
description: Parse a PDF into Markdown for Agent/LLM consumption
---

Parse a user-specified PDF into structured Markdown.


# How to Use

Try **MinerU** first. If it fails, fall back to **pymupdf4llm + repair**.


## Errors

- Analyze the error message.
- If any config error, prompt user to configure interactively.
- Otherwise, print the error message and exit.


## Sub-skills

- **MinerU** — cloud-based, high-quality parsing with formulas, tables, and images → `sub_skills/mineru/skill.md`
- **pymupdf4llm + repair** — local parsing with LLM-based structural repair for double-column papers → `sub_skills/pymupdf4llm/skill.md`
