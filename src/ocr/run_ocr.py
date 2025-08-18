from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import time
import json
import shutil

import pandas as pd
import yaml

# Optional deps
try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


def _load_cfg() -> dict:
    with open("configs/app.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\xa0", " ")).strip()


def _tesseract_setup(cfg: dict) -> None:
    if pytesseract is None:
        return
    tess_cmd = cfg.get("ocr", {}).get("tesseract_cmd")
    if tess_cmd:
        pytesseract.pytesseract.tesseract_cmd = tess_cmd


def _tesseract_available() -> bool:
    # respect explicit path or PATH
    if pytesseract is None:
        return False
    cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", None)
    if cmd and Path(cmd).exists():
        return True
    return shutil.which("tesseract") is not None


_missing_warned = set()

def _pytess(img_path: Path, langs: list[str], psm: int) -> str:
    if pytesseract is None or Image is None:
        return ""
    for chain in ["+".join(langs), "rus", "kaz", "eng"]:
        try:
            with Image.open(img_path) as im:
                return _norm_ws(
                    pytesseract.image_to_string(im, lang=chain, config=f"--oem 1 --psm {psm}")
                )
        except Exception as e:
            msg = str(e)
            if "Error opening data file" in msg or "Failed loading language" in msg:
                if chain not in _missing_warned:
                    _missing_warned.add(chain)
                    print(f"[ocr] WARNING: tesseract language '{chain}' not available; will try fallback")
            continue
    return ""



def _render_page(pdf_path: Path, page_number: int, dpi: int, out_dir: Path) -> Optional[Path]:
    if fitz is None:
        return None
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with fitz.open(pdf_path) as doc:
            p = doc.load_page(page_number - 1)
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = p.get_pixmap(matrix=mat, alpha=False)
            out = out_dir / f"{pdf_path.stem}_p{page_number}.png"
            pix.save(out.as_posix())
            return out
    except Exception:
        return None


def _find_pdf(doc_name: str, roots: List[Path]) -> Optional[Path]:
    for root in roots:
        cand = root / doc_name
        if cand.exists():
            return cand
    for root in roots:  # fallback rglob
        for p in root.rglob(doc_name):
            return p
    return None


def run_ocr(cache_dir: str) -> None:
    cfg = _load_cfg()
    _tesseract_setup(cfg)

    text_min_len = int(cfg.get("ingest", {}).get("text_min_len", 80))
    ocr_cfg = cfg.get("ocr", {})
    lang_chain = "+".join(ocr_cfg.get("languages", ["rus", "kaz"]))
    psm_A = int(ocr_cfg.get("profiles", {}).get("A", {}).get("psm", 6))
    psm_B = int(ocr_cfg.get("profiles", {}).get("B", {}).get("psm", 4))
    dpi = int(ocr_cfg.get("dpi", 300))

    have_tess = _tesseract_available()
    if not have_tess:
        print("[ocr] WARNING: Tesseract not found (PATH or ocr.tesseract_cmd). Scan pages will remain empty.")

    cache = Path(cache_dir)
    tmp_img = cache / "ocr_renders"
    pages_path = cache / "pages.parquet"
    out_path = cache / "page_text.parquet"

    if not pages_path.exists():
        print(f"[ocr] {pages_path} missing -> writing empty")
        pd.DataFrame(columns=["doc_name", "page_number", "text_final", "profile_used"]).to_parquet(out_path, index=False)
        return

    pages = pd.read_parquet(pages_path)
    if pages.empty:
        print("[ocr] pages empty -> writing empty")
        pd.DataFrame(columns=["doc_name", "page_number", "text_final", "profile_used"]).to_parquet(out_path, index=False)
        return

    # Optional whitelist from answers_public.json (speed-up): still OCR image-scan even if not listed.
    wl: set[Tuple[str, int]] = set()
    try:
        with open("data/public/answers_public.json", "r", encoding="utf-8") as f:
            ans = json.load(f)
        for a in ans:
            for rc in a.get("relevant_chunks", []):
                wl.add((rc["document_name"], int(rc["page_number"])))
        if wl:
            print(f"[ocr] whitelist detected: {len(wl)} doc-page pairs (used only to skip extra text-layer pages)")
    except Exception:
        pass

    corpus_roots = [Path(cfg.get("paths", {}).get("corpus_root", "data/corpus_flat")), Path("data/corpus")]

    t0 = time.time()
    total = len(pages)
    text_nonempty = scan_nonempty = 0
    rows: List[Dict] = []

    for i, r in enumerate(pages.itertuples(index=False), 1):
        doc = r.doc_name
        pno = int(r.page_number)
        media = getattr(r, "media_type", "")
        text_raw = getattr(r, "text_raw", "") or ""
        img_hint = Path(getattr(r, "image_path", "") or "")

        # Speed: if whitelist exists and page is NOT listed:
        #   - pass through text-layer pages quickly
        #   - DO NOT skip image-scan pages (we still OCR them!)
        if wl and (doc, pno) not in wl and media == "text-layer":
            if len(text_raw) >= text_min_len:
                rows.append({"doc_name": doc, "page_number": pno, "text_final": text_raw, "profile_used": "pass-through"})
                text_nonempty += 1
            else:
                rows.append({"doc_name": doc, "page_number": pno, "text_final": "", "profile_used": "skip"})
            if i % 200 == 0 or i == total:
                print(f"[ocr] {i}/{total} pages | text_nonempty={text_nonempty} | scan_nonempty={scan_nonempty} | elapsed={time.time()-t0:.1f}s")
            continue

        # Normal path
        if media == "text-layer" and len(text_raw) >= text_min_len:
            rows.append({"doc_name": doc, "page_number": pno, "text_final": text_raw, "profile_used": "pass-through"})
            text_nonempty += 1
        elif media in ("image-scan", "corrupt-font"):
            txt = ""
            profile = "skip"
            if have_tess:
                # 1) use preview image if present
                if img_hint.exists():
                    txt = _pytess(img_hint, ocr_cfg.get("languages", ["rus","kaz"]), psm_A) or _pytess(img_hint, ocr_cfg.get("languages", ["rus","kaz"]), psm_B)
                # 2) render from PDF if still empty or no preview
                if not txt:
                    pdf_path = _find_pdf(doc, corpus_roots)
                    if pdf_path:
                        img = _render_page(pdf_path, pno, dpi, tmp_img)
                        if img and img.exists():
                            txt = _pytess(img, lang_chain, psm_A) or _pytess(img, lang_chain, psm_B)
                            profile = "A*" if txt else profile
            rows.append({"doc_name": doc, "page_number": pno, "text_final": txt, "profile_used": f"tesseract-{profile}"})
            if txt:
                scan_nonempty += 1
        else:
            rows.append({"doc_name": doc, "page_number": pno, "text_final": "", "profile_used": "skip"})

        if i % 200 == 0 or i == total:
            print(f"[ocr] {i}/{total} pages | text_nonempty={text_nonempty} | scan_nonempty={scan_nonempty} | elapsed={time.time()-t0:.1f}s")

    out = pd.DataFrame(rows)[["doc_name", "page_number", "text_final", "profile_used"]]
    out.to_parquet(out_path, index=False)
    print(f"[ocr] wrote {out_path} rows={len(out)} | non-empty total={(out.text_final.str.len()>0).sum()} "
          f"(text-layer={text_nonempty}, scan={scan_nonempty})")
