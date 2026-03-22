from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class ParseResult:
    markdown: str
    images: list[Path] = field(default_factory=list)
    source: Literal["mineru", "pymupdf4llm"] = "pymupdf4llm"
    pages: int = 0
    out_dir: Path | None = None
