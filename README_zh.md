# pdf-skill

将学术论文 PDF 解析为干净 Markdown 的工具，专为 Agent / LLM 流程设计。

[English](README.md)

## 功能

- **MinerU API**：精准解析，保留公式（LaTeX）、表格（HTML）、图片
- **pymupdf4llm**：本地解析，无网络依赖
- **LLM 修复**：针对双栏 PDF 的结构问题（标题空格、断词、乱序、脚注、图注残留等），四阶段流水线清洗

## 安装

```bash
pip install -e .
```

依赖：`pymupdf4llm`、`pymupdf`、`requests`、`anthropic`

## 配置

各 sub_skill 独立配置，首次运行时自动提示填写。也可直接编辑：

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

支持任何兼容 Anthropic Messages API 的 LLM 服务。也可通过环境变量覆盖（`MINERU_API_KEY`、`LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`）。

## 用法

### MinerU 解析

```python
from sub_skills.mineru import MinerUParser
from sub_skills.mineru.config import load_config
from pathlib import Path

cfg = load_config()
result = MinerUParser(cfg).parse(Path("paper.pdf"), Path("output/"))
print(result.markdown)  # 完整 Markdown
print(result.images)    # [Path, ...]
```

### pymupdf4llm 本地解析

```python
from sub_skills.pymupdf4llm import FallbackParser
from pathlib import Path

result = FallbackParser().parse(Path("paper.pdf"), Path("output/"))
print(result.markdown)
```

### LLM 修复双栏 Markdown

```python
from sub_skills.pymupdf4llm import repair, repair_file

# 修复字符串
fixed_md = repair(raw_markdown, verbose=True)

# 修复文件（输出为 <stem>_repaired.md）
out_path = repair_file("output/full.md", verbose=True)
```

修复参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | config.json / `claude-3-5-haiku-20241022` | LLM 模型 |
| `max_chars_per_chunk` | `3000` | 每次 LLM 调用的最大字符数 |
| `seam_context` | `2` | 边界处理时每侧取的段落数 |
| `max_workers` | `4` | 并发线程数 |

## 项目结构

```
sub_skills/
  mineru/
    config.json      # MinerU API Key 配置
    config.py        # 配置加载
    models.py        # ParseResult 数据类
    mineru.py        # MinerU API 客户端
    skill.md         # 调用说明
  pymupdf4llm/
    config.json      # LLM 配置
    config.py        # 配置加载
    models.py        # ParseResult 数据类
    fallback.py      # 本地 PDF 解析
    repair.py        # LLM 修复流水线
    pdf_repair.md    # 修复流程设计文档
    skill.md         # 调用说明
skill.md             # 顶层调度说明（自然语言）
```
