---
name: pdf-mineru
description: 通过 MinerU API 将 PDF 解析为结构化 Markdown（精准，含公式/表格/图片）
---

## 前置条件

`config.json` 中须配置 `mineru_api_key`（或环境变量 `MINERU_API_KEY`）。

## 调用

```python
from sub_skills.mineru import MinerUParser
from sub_skills.config import load_config
from pathlib import Path

cfg = load_config()
result = MinerUParser(cfg).parse(Path("<pdf_path>"), Path("<out_dir>"))
# result.markdown  — 完整 Markdown（含公式 LaTeX、表格 HTML）
# result.images    — [Path, ...]  提取的图片文件
# result.pages     — 页数
# result.out_dir   — 输出目录（含 full.md、images/、result.zip）
```

## API 流程

1. `POST /file-urls/batch` — 获取上传 URL
2. `PUT <signed_url>` — 上传 PDF
3. `POST /extract/task/batch` — 提交解析任务（启用公式、表格、OCR）
4. `GET /extract-results/batch/{id}` — 轮询（每 5 秒，最长 10 分钟）
5. 下载 `full_zip_url` → 解压至输出目录

## 耗时

约 30–120 秒，取决于页数。

## 错误处理

- API Key 缺失 → 抛出异常，提示用户配置 API Key
- 网络超时 / 解析失败 → 抛出异常，提示用户相关错误
