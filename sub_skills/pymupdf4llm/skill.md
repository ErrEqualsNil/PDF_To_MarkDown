---
name: pdf-pymupdf4llm
description: 本地 PDF 解析（pymupdf4llm）+ 双栏学术论文 LLM 结构修复
---

## 解析

无需网络，本地直接解析。

```python
from sub_skills.pymupdf4llm import FallbackParser
from pathlib import Path

result = FallbackParser().parse(Path("<pdf_path>"), Path("<out_dir>"))
# result.markdown  — 原始 Markdown
# result.images    — 提取的图片文件
# result.out_dir   — 输出目录（含 full.md、images/）
```

双栏排版的学术论文会产生结构问题（标题空格、断词、乱序、脚注混入等），建议解析后执行修复。

## 修复

`config.json` 中须配置 `llm_api_key` 和 `llm_base_url`（支持任何兼容 Anthropic Messages API 的服务）。

```python
from sub_skills.pymupdf4llm import repair, repair_file

# 修复字符串
fixed_md = repair(raw_markdown, verbose=True)

# 修复文件，输出为 <stem>_repaired.md（同目录）
out_path = repair_file("output/full.md", verbose=True)
```

### 详细设计

仅在错误排查时，参考 `pdf_repair.md`。

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `api_key` | config.json | LLM API Key |
| `base_url` | config.json | LLM Base URL |
| `model` | config.json / `claude-3-5-haiku-20241022` | LLM 模型 |
| `max_chars_per_chunk` | `3000` | 每次调用最大字符数 |
| `seam_context` | `2` | 边界拼接时每侧取的段落数 |
| `max_workers` | `4` | 并发线程数 |

详细设计见 `pdf_repair.md`。

## 错误处理

- API Key 缺失 → 抛出异常，提示用户配置 API Key
- 网络超时 / 解析失败 → 抛出异常，提示用户相关错误
