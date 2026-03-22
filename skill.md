---
name: pdf
description: 将 PDF 解析为 Markdown，供 Agent/LLM 消费
---

解析用户指定的 PDF 文件为结构化 Markdown。

## Sub-skills

- **MinerU 解析**：详见 `sub_skills/mineru/skill.md`
- **pymupdf4llm 解析 + 修复**：详见 `sub_skills/pymupdf4llm/skill.md`

## 解析策略

优先使用 **MinerU 解析**

错误回退 **pymupdf4llm 解析 + 修复**

