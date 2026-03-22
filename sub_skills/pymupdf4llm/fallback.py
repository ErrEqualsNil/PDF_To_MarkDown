"""
pymupdf4llm fallback parser — local, no network required.
Outputs Markdown with image placeholders.
"""

from __future__ import annotations

from pathlib import Path

from .models import ParseResult


class FallbackParser:
    def parse(self, pdf_path: Path, out_dir: Path) -> ParseResult:
        import pymupdf4llm
        import fitz

        md_text = pymupdf4llm.to_markdown(str(pdf_path), write_images=True,
                                          image_path=str(out_dir / "images"))
        doc    = fitz.open(str(pdf_path))
        pages  = len(doc)
        doc.close()

        md_path = out_dir / pdf_path.name.replace(pdf_path.suffix, ".md")
        md_path.write_text(md_text, encoding="utf-8")

        img_files = list((out_dir / "images").rglob("*"))
        img_files = [f for f in img_files if f.suffix in (".png", ".jpg", ".jpeg")]

        return ParseResult(markdown=md_text, images=img_files,
                           pages=pages, out_dir=out_dir)
