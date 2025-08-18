import json
from pathlib import Path
from typing import List

import pandas as pd
import yaml
from PIL import Image

try:
    from PyPDF2 import PdfReader
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None


def _load_app_config() -> dict:
    with open("configs/app.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _scan_pdf(pdf_path: Path, cfg: dict) -> List[dict]:
    """Return list of page records for a single PDF."""
    records: List[dict] = []
    if PdfReader is None:
        # fallback: single empty page
        text = ""
        records.append(
            {
                "doc_name": pdf_path.name,
                "page_number": 1,
                "media_type": "image-scan",
                "text_raw": None,
                "image_path": str(pdf_path.with_suffix(".png")),
                "risky_page": True,
            }
        )
        return records

    reader = PdfReader(str(pdf_path))
    ingest_cfg = cfg.get("ingest", {})
    text_min_len = ingest_cfg.get("text_min_len", 0)
    badchars_max_ratio = ingest_cfg.get("badchars_max_ratio", 1.0)
    corrupt_font_ratio = ingest_cfg.get("corrupt_font_ratio", 1.0)

    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text_len = len(text)
        bad_ratio = text.count("�") / max(text_len, 1)
        corrupt_ratio = 0.0  # PyPDF2 can't detect, assume 0
        media_type = "text-layer"
        risky = False
        if text_len < text_min_len:
            media_type = "image-scan"
        elif bad_ratio > badchars_max_ratio or corrupt_ratio >= corrupt_font_ratio:
            media_type = "corrupt-font"
            risky = True
        text_raw = text if media_type == "text-layer" else None

        image_path = None
        if media_type != "text-layer":
            # create blank preview placeholder
            preview_dir = Path("data/cache/previews") / pdf_path.stem
            preview_dir.mkdir(parents=True, exist_ok=True)
            image_path = preview_dir / f"{idx+1}.png"
            if not image_path.exists():
                Image.new("RGB", (10, 10), color="white").save(image_path)

        records.append(
            {
                "doc_name": pdf_path.name,
                "page_number": idx + 1,
                "media_type": media_type,
                "text_raw": text_raw,
                "image_path": str(image_path) if image_path else None,
                "risky_page": risky,
            }
        )

    return records


def ingest_pdfs(corpus_dir: str, out_dir: str) -> None:
    cfg = _load_app_config()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    corpus = Path(corpus_dir)
    all_records: List[dict] = []
    summary_rows = []
    for pdf in sorted(corpus.glob("*.pdf")):
        pages = _scan_pdf(pdf, cfg)
        all_records.extend(pages)
        summary_rows.append(
            {
                "doc": pdf.name,
                "pages_total": len(pages),
                "text_layer_pages": sum(p["media_type"] == "text-layer" for p in pages),
                "image_scan_pages": sum(p["media_type"] != "text-layer" for p in pages),
                "corrupt_flags": sum(p["risky_page"] for p in pages),
            }
        )

    df = pd.DataFrame(all_records)
    if not df.empty:
        df.to_parquet(out_path / "pages.parquet", index=False)
    else:
        df.to_parquet(out_path / "pages.parquet", index=False)

    pd.DataFrame(summary_rows).to_csv("logs/ingest_summary.csv", index=False)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", dest="out", required=True)
    args = parser.parse_args()
    ingest_pdfs(args.inp, args.out)
