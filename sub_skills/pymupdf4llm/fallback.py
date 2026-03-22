"""
pymupdf4llm fallback parser — local, no network required.
Outputs Markdown to <out_dir>/<pdf_name>.md.
"""

from __future__ import annotations

from pathlib import Path

from .models import ParseResult


class FallbackParser:
    def parse(self, pdf_path: Path, out_dir: Path) -> ParseResult:
        import pymupdf4llm
        import fitz
        out_dir.mkdir(parents=True, exist_ok=True)

        md_text = pymupdf4llm.to_markdown(str(pdf_path), write_images=False)
        doc    = fitz.open(str(pdf_path))
        pages  = len(doc)
        doc.close()

        md_path = out_dir / pdf_path.name.replace(pdf_path.suffix, ".md")
        md_path.write_text(md_text, encoding="utf-8")

        return ParseResult(markdown=md_text, pages=pages, out_dir=out_dir)
