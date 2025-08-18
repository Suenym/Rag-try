from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import re

import fitz  # PyMuPDF
import pandas as pd
import yaml


PAGES_COLS = ["doc_name", "page_number", "media_type", "text_raw", "image_path", "risky_page"]


def _load_cfg() -> dict:
    with open("configs/app.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _badchars_ratio(s: str) -> float:
    if not s:
        return 0.0
    # crude proxy: replacement char + control chars (except \n\t)
    bad = sum(ch == "\uFFFD" or (ord(ch) < 32 and ch not in ("\n", "\t")) for ch in s)
    return bad / max(1, len(s))


def ingest_pdfs(in_dir: str, out_dir: str) -> None:
    cfg = _load_cfg()
    text_min_len = int(cfg.get("ingest", {}).get("text_min_len", 80))
    badchars_max = float(cfg.get("ingest", {}).get("badchars_max_ratio", 0.02))

    in_path = Path(in_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    prev_dir = out_path / "previews"
    prev_dir.mkdir(parents=True, exist_ok=True)

    pdfs = list(in_path.rglob("*.pdf"))
    print(f"[ingest] PDFs found: {len(pdfs)} under {in_path}")

    rows: List[Dict] = []
    for pdf in pdfs:
        try:
            doc = fitz.open(pdf)
        except Exception as e:
            print(f"[ingest] skip {pdf.name}: open error {e}")
            continue

        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            # normalize whitespace minimally so lengths are meaningful
            text_norm = re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()
            br = _badchars_ratio(text_norm)

            if len(text_norm) >= text_min_len and br <= badchars_max:
                media = "text-layer"
                image_path = ""
            else:
                media = "image-scan"
                # save a preview to help OCR/debug
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    png = prev_dir / f"{pdf.stem}_p{i+1}.png"
                    pix.save(png.as_posix())
                    image_path = str(png)
                except Exception:
                    image_path = ""

            rows.append(
                {
                    "doc_name": pdf.name,
                    "page_number": i + 1,  # 1-based
                    "media_type": media,
                    "text_raw": text_norm if media == "text-layer" else "",
                    "image_path": image_path,
                    "risky_page": False,
                }
            )
        doc.close()

    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(columns=PAGES_COLS)

    # enforce schema order
    for c in PAGES_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[PAGES_COLS]

    df.to_parquet(out_path / "pages.parquet", index=False)
    print(f"[ingest] wrote pages.parquet rows={len(df)}")
