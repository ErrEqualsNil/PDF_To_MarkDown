# PDF 结构修复设计文档

针对 `pymupdf4llm` 解析双栏学术论文 PDF 产生的 Markdown 文本，进行结构性修复的设计说明。

---

## 一、问题类型

| # | 问题 | 示例 |
|---|------|------|
| 1 | 章节标题字符间被注入空格 | `II. T HE O VERALL P IPELINE` |
| 2 | 脚注插入正文段落中间 | `1 Tested with Faiss-IndexFlat on our workstation.` |
| 3 | 作者/机构信息散落正文 | `5th Yongheng Liu Pengcheng Laboratory Shenzhen, China` |
| 4 | 图片引用和说明文字残留 | `![](images/xxx.png)` / `Fig. 1: Retrieval is...` |
| 5 | 表格被解析为管道行或文本 | `\|.8\|Col2\|` / `**Symbol** **Description** q Current...` |
| 6 | 数学符号被下划线包裹 | `_q_`、`_C_`、`_D_` |
| 7 | 双栏断词合并 | `domainspecific`、`reusebased` |
| 8 | 双栏读取顺序错乱 | 左右栏交替，句子被截断后插入另一段 |
| 9 | 序数括号格式 | `1 [st]`、`2 [nd]`、`3 [rd]` |
| 10 | 段落内大量多余换行 | 同一段落被拆成 10+ 行 |
| 11 | 公式渲染为散乱文本 | 孤立希腊字母、等号、分数符号散落段落中 |
| 12 | 图例/图表碎片标签残留 | `(a) Approximate Nearest Neighbor Search` |

---

## 二、修复流程

```
输入: pymupdf4llm 原始 Markdown
   ↓
① 预处理（纯规则）
   删除图片/表格/说明文字、元信息段落
   修复序数/下划线、合并软换行、过滤碎片段落
   ↓
② 分块（按段落/章节边界，不截断段落）
   ↓
③ LLM 修复（并发，每块独立调用）
   修复标题/断词/乱序，删除脚注/作者信息，标记公式
   ↓
④ 规则拼接（相邻块边界 seam 处理）
   检测末段句子完整性，不完整则直接拼接下块首段
   ↓
输出: 结构整洁的 Markdown
```

---

## 三、预处理（`preprocess`）

### 3.1 删除图片、表格及说明

逐行扫描，维护两个状态机：

- **`in_table`**：检测到 `**Word** **Word**` 格式的表格标题后激活，持续丢弃后续短行（< 20 词）。两种情况退出：①遇到词数 ≥ 20 的正文长句；②遇到列表项（`1)` / `-` / `•` / `#` 开头），避免误删编号列表。
- **`in_caption`**：检测到 `Fig. N` / `Figure N` / `TABLE N` 行后激活，持续丢弃后续非空行，直到空行为止（处理图注多行续文）。

同时删除：`![...](...)` 图片行、`|...|` 管道表格行。

### 3.2 删除元信息段落（`_remove_meta_blocks`）

在 `_merge_soft_wraps` **之后**执行（先合并软换行，才能将多行的基金段落识别为完整段落）。对每个段落块匹配以下前缀，命中则整段丢弃：
- `Corresponding Author:`
- `This work was supported by`
- `The authors acknowledge`

### 3.3 文本修复

- **序数**：`1 [st]` → `1st`，`2 [nd]` → `2nd`，`3 [rd]` → `3rd`
- **下划线数学符号**：`_q_` → `q`（单字符/短 token 斜体标记）

### 3.4 合并段落内软换行

以 `\n\n` 为段落边界，将每段内所有行合并为一行。标题行（`#` 开头）和代码块（` ``` ` 包裹）保持原样。

### 3.5 过滤过短段落

删除词数 < 6 的段落块。**例外保留**：`#` 开头的标题、`1)` / `-` / `•` 开头的列表项、`[N]` 开头的参考文献。

---

## 四、分块策略（`split_chunks`）

**原则：不在段落中间截断。** 以段落为最小单位分组。

分块规则（优先级从高到低）：

1. **章节标题前强制分块** — 遇到 `I.`、`II.`、`A.` 等标题时，若当前块 > 300 字符则先输出
2. **长度超限时分块** — 加入下一段落后超过 `max_chars`（默认 3000）则先输出
3. **单段落超限时独立成块** — 单段落超限时独立作为一块

---

## 五、LLM 修复（`repair_chunk`）

**模型**：MiniMax-M2.7（通过 Anthropic SDK 兼容接口，`base_url = https://api.minimaxi.com/anthropic`）

| 任务 | 说明 |
|------|------|
| 章节标题重建 | `II. T HE O VERALL P IPELINE` → `## II. The Overall Pipeline` |
| 脚注删除 | 以数字开头的短行，如 `1 Tested with Faiss…` |
| 作者/机构信息清除 | 正文中夹杂的姓名、大学、邮箱、城市行 |
| 断词修复 | `domainspecific` → `domain-specific` |
| 双栏乱序修复 | 重新排列被另一栏打断的句子 |
| 公式文本标记 | 孤立符号包裹在 ` ```math ``` ` 块中 |

**约束**：只输出修复后文本，不加解释；保留引用标记 `[1]`、`[2]–[4]`；不补充原文不存在的内容；**严禁丢弃编号列表项**（`1)` / `2)` 等）；删除孤立介词/副词碎片（图注续行遗留，如 `by the comparison between…`）。

---

## 六、规则拼接（`merge_chunk`）

双栏交汇处段落常被截断在相邻 chunk 之间：

```
Chunk A 末尾: "...ANNS, such as IVF and ScaNN [10], are"
Chunk B 开头: "well-established in industry, as shown in Fig.2a."
```

**实现**（纯规则，LLM 有概率丢弃段落或拒绝合并，故弃用）：

取相邻两块边界各 `seam_context`（默认 2）个段落，对 tail 末段执行：

1. 末字符不属于 `.!?。！？` → 句子未完
2. 将 tail 末段与 head 首段用空格拼接，从 head 列表移除首段
3. head 首段首字母若为大写则改小写（处于句中而非句首）
4. 回写时用原始 `len(head)` 而非 `len(new_head)` 作为切片上界（`repaired[i+1][:len(head)] = new_head`），确保合并的首段从下一 chunk 中实际移除；否则合并段会同时出现在两个 chunk 中，造成内容重复

---

## 七、已知局限性

以下问题经测试确认存在，但影响可接受：

| 问题 | 原因 | 状态 |
|------|------|------|
| Introduction 末段可能游离至 Section II 开头 | 双栏乱序跨 chunk，LLM 无法感知前后 chunk 内容 | 可接受 |
| 算法伪代码格式不一致（列表与代码块混用） | LLM 对无固定格式的算法块格式化结果不稳定 | 可接受 |

---

## 八、调用方式

```python
from sub_skills.pdf import parse
from sub_skills.pymupdf4llm import repair, repair_file

result = parse("paper.pdf")                          # MinerU 首选，pymupdf4llm fallback
fixed_md = repair(raw_markdown, verbose=True)        # 单独修复 Markdown 字符串
out_path = repair_file("output/full.md", verbose=True)  # 修复文件
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | `MiniMax-M2.7` | LLM 模型名 |
| `max_chars_per_chunk` | `3000` | 每次 LLM 调用的最大字符数 |
| `seam_context` | `2` | 每侧参与 seam 验证的段落数 |
| `verbose` | `False` | 打印进度信息 |
