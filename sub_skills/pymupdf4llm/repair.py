"""
PDF Markdown 结构修复 — 详见 pdf_repair.md

四阶段流程：
  ① preprocess  — 规则清洗（无 LLM）
  ② split_chunks — 段落感知分块
  ③ repair_chunk — LLM 并发修复每个 chunk
  ④ merge_chunk  — 规则整理相邻 chunk 边界
"""

from __future__ import annotations

import re
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# ① 预处理
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    text = _remove_media(text)
    text = re.sub(r"(\d+)\s*\[(st|nd|rd|th)\]", lambda m: m.group(1) + m.group(2), text, flags=re.I)
    text = re.sub(r"_([A-Za-z0-9α-ω]{1,4})_", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _merge_soft_wraps(text)
    text = _remove_meta_blocks(text)
    text = _drop_short_blocks(text, min_words=6)
    return text


def _remove_media(text: str) -> str:
    """删除图片、表格行、说明文字、及非管道格式的表格残留内容。"""
    out, in_table, in_caption = [], False, False
    for line in text.splitlines():
        s = line.strip()
        # 图注续行（caption 未结束）：遇到空行才退出
        if in_caption:
            if not s:
                in_caption = False   # 空行：退出 caption 模式，保留空行
            else:
                continue             # 非空行：caption 续行，删除
        if s.startswith("![") or s.startswith("|"):
            in_table = False
            continue
        if re.match(r"^(Fig\.|Figure|TABLE|Table)\s*[\dIVX]+", s):
            in_table = False
            in_caption = True        # 可能有续行，保持删除状态
            continue
        # 非管道表格标题：**Word** **Word**
        if re.match(r"^\*\*\S+\*\*\s+\*\*\S+\*\*", s):
            in_table = True
            continue
        # 表格数据行（< 20 词）；遇到列表项或标题时强制退出表格模式
        if in_table:
            if re.match(r"^(\d+[.)]\s|[-*•]\s|#)", s):
                in_table = False          # 列表项/标题，保留并退出表格模式
            elif len(s.split()) < 20:
                continue                  # 短行，视为表格数据行，丢弃
            else:
                in_table = False          # 长正文句，退出表格模式
        out.append(line)
    return "\n".join(out)


def _merge_soft_wraps(text: str) -> str:
    """将每个段落块内的软换行合并为单行；标题和代码块保持不变。"""
    merged = []
    for block in re.split(r"\n{2,}", text):
        first = block.lstrip().split("\n")[0].lstrip()
        if first.startswith("```") or first.startswith("#"):
            merged.append(block)
        else:
            merged.append(" ".join(l.strip() for l in block.splitlines() if l.strip()))
    return "\n\n".join(merged)


def _drop_short_blocks(text: str, min_words: int) -> str:
    """删除词数不足的碎片段落；保留标题、列表项、参考文献条目。"""
    kept = []
    for block in re.split(r"\n{2,}", text):
        s = block.strip()
        if not s:
            continue
        if (s.startswith("#")
                or re.match(r"^[-*•]\s|^\d+[.)]\s", s)
                or re.match(r"^\[\d+\]", s)
                or len(s.split()) >= min_words):
            kept.append(block)
    return "\n\n".join(kept)


_META_RE = re.compile(
    r"^(Corresponding Author|This work was supported|The authors? (gratefully )?acknowledge)",
    re.I,
)


def _remove_meta_blocks(text: str) -> str:
    """删除致谢、基金资助、通讯作者等首页夹带的元信息段落。"""
    kept = []
    for block in re.split(r"\n{2,}", text):
        if not _META_RE.search(block.strip()):
            kept.append(block)
    return "\n\n".join(kept)


# ─────────────────────────────────────────────────────────────────────────────
# ② 分块
# ─────────────────────────────────────────────────────────────────────────────

_SECTION_RE = re.compile(r"^(?:[IVX]+\.|[A-Z]\.|Algorithm\s+\d+)\s+[A-Z]")


def split_chunks(text: str, max_chars: int = 3000) -> list[str]:
    """段落感知分块：不在段落中间截断，优先在章节标题处开新块。"""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, current, cur_len = [], [], 0

    for para in paragraphs:
        is_section = bool(_SECTION_RE.match(para))
        if (is_section and current and cur_len > 300) or \
           (current and cur_len + len(para) + 2 > max_chars):
            chunks.append("\n\n".join(current))
            current, cur_len = [], 0
        current.append(para)
        cur_len += len(para) + 2

    if current:
        chunks.append("\n\n".join(current))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# LLM 调用基础
# ─────────────────────────────────────────────────────────────────────────────

def _llm(client, model: str, system: str, user: str) -> str:
    resp = client.messages.create(
        model=model, max_tokens=8192, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return next((b.text.strip() for b in resp.content if b.type == "text"), "")


# ─────────────────────────────────────────────────────────────────────────────
# ③ LLM 修复
# ─────────────────────────────────────────────────────────────────────────────

_REPAIR_SYSTEM = textwrap.dedent("""\
    你是一个学术 PDF 文本修复助手。输入是 pymupdf4llm 解析双栏 PDF 得到的原始 Markdown，
    存在结构性问题。请返回修复后的纯文本，不加任何解释或前言。

    修复规则：
    1. 章节标题：如 "II. T HE O VERALL P IPELINE" 是解析器注入了空格，
       重建为正确标题并格式化：罗马数字节 → "## II. Title"，字母子节 → "### A. Title"。
    2. 脚注：删除以数字开头的短说明行，如 "1 Tested with Faiss…"、"2 Data source: …"。
    3. 作者/机构信息：删除正文中夹杂的姓名、大学名、邮箱、城市/国家行。
    4. 断词修复：将双栏换行导致的合并词还原，如 "domainspecific" → "domain-specific"。
    5. 双栏乱序：若句子明显被另一栏内容打断，重新排列为正确阅读顺序。
    6. 公式文本：孤立的希腊字母、等号、分数等公式碎片，包裹在 ```math … ``` 中。
    7. 保留所有引用标记 [1]、[2]–[4]，不补充输入中不存在的内容。
    8. 严禁丢弃内容：以 "1)"/"2)"/"3)" 等数字加括号开头的列表项必须完整保留；
       开头明显是上一句延续（无主语、以小写或连词开头）的段落，须与前一段合并，不得删除。
    9. 删除孤立碎片段落：以介词/副词/小写字母开头、明显是图注或标题残留的独立短段落（如
       "by the comparison between…"、"as shown in Fig."），不属于正文句子，应删除。
""")


def repair_chunk(chunk: str, client, model: str) -> str:
    return _llm(client, model, _REPAIR_SYSTEM, chunk)


# ─────────────────────────────────────────────────────────────────────────────
# ④ 规则整理（chunk 边界合并）
# ─────────────────────────────────────────────────────────────────────────────

_SENTENCE_END = re.compile(r'[.!?。！？]["\']?\s*$')


def merge_chunk(tail: list[str], head: list[str]) -> tuple[list[str], list[str]]:
    """规则合并：若 tail 末段句子未完（不以句终符结尾），直接拼接 head 首段。"""
    if not tail or not head:
        return tail, head
    if not _SENTENCE_END.search(tail[-1]):
        continuation = head[0].lstrip()
        # 拼接处的首字母若大写，改为小写（它是句子中间，而非新句开头）
        if continuation and continuation[0].isupper():
            continuation = continuation[0].lower() + continuation[1:]
        merged = tail[-1].rstrip() + " " + continuation
        return tail[:-1] + [merged], head[1:]
    return tail, head


# ─────────────────────────────────────────────────────────────────────────────
# 公开 API
# ─────────────────────────────────────────────────────────────────────────────

def repair(
    markdown: str,
    *,
    api_key: str = "",
    base_url: str = "",
    model: str = "",
    max_chars_per_chunk: int = 3000,
    seam_context: int = 2,
    max_workers: int = 4,
    verbose: bool = False,
) -> str:
    """
    修复 pymupdf4llm 输出的 Markdown。

    Args:
        markdown:            原始 Markdown 字符串。
        api_key:             LLM API Key（也可在 config.json llm_api_key 中配置）。
        base_url:            LLM Base URL（也可在 config.json llm_base_url 中配置）。
        model:               使用的模型名称（也可在 config.json llm_model 中配置）。
        max_chars_per_chunk: 每次 LLM 修复调用的最大字符数。
        seam_context:        边界整理时每侧取的段落数。
        max_workers:         并发修复的最大线程数。
        verbose:             打印进度。
    """
    import anthropic
    from .config import load_config

    cfg      = load_config()
    key      = api_key  or cfg.get("llm_api_key", "")
    base_url = base_url or cfg.get("llm_base_url", "") or None
    model    = model    or cfg.get("llm_model", "") or "claude-3-5-haiku-20241022"
    if not key:
        raise ValueError(
            "未找到 LLM API Key，请在 config.json 中设置 llm_api_key，"
            "或通过环境变量 LLM_API_KEY 传入。"
        )

    client = anthropic.Anthropic(api_key=key, base_url=base_url)

    # ① 预处理
    if verbose:
        print("  [1/3] 预处理...", flush=True)
    cleaned = preprocess(markdown)

    # ② 分块
    chunks = split_chunks(cleaned, max_chars=max_chars_per_chunk)
    if verbose:
        print(f"  [2/3] 并发修复 {len(chunks)} 个 chunk（workers={max_workers}）...", flush=True)

    # ③ 并发 LLM 修复
    repaired: list[list[str]] = [[] for _ in chunks]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(repair_chunk, chunk, client, model): i
                   for i, chunk in enumerate(chunks)}
        for fut in as_completed(futures):
            i = futures[fut]
            paras = [p.strip() for p in re.split(r"\n{2,}", fut.result()) if p.strip()]
            repaired[i] = paras
            if verbose:
                print(f"        chunk {i+1}/{len(chunks)} 完成（{len(paras)} 段）", flush=True)

    # ④ 规则整理（chunk 边界合并）
    if len(repaired) > 1:
        if verbose:
            print(f"  [3/3] 整理 {len(repaired)-1} 处边界...", flush=True)
        for i in range(len(repaired) - 1):
            tail = repaired[i][-seam_context:]
            head = repaired[i + 1][:seam_context]
            if not tail or not head:
                continue
            new_tail, new_head = merge_chunk(tail, head)
            repaired[i][-len(tail):]    = new_tail
            repaired[i + 1][:len(head)] = new_head
            if verbose:
                print(f"        边界 {i+1} 完成", flush=True)
    result = "\n\n".join(p for chunk_paras in repaired for p in chunk_paras)
    return re.sub(r"\n{3,}", "\n\n", result)


def repair_file(
    md_path: str | Path,
    out_path: str | Path | None = None,
    **kwargs,
) -> Path:
    """修复磁盘上的 .md 文件，返回输出路径。"""
    md_path = Path(md_path)
    fixed   = repair(md_path.read_text(encoding="utf-8"), **kwargs)
    out     = Path(out_path) if out_path else md_path.parent / (md_path.stem + "_repaired.md")
    out.write_text(fixed, encoding="utf-8")
    return out
