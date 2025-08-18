from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import yaml


def _load_configs() -> tuple[dict, dict]:
    with open("configs/app.yaml", "r", encoding="utf-8") as f:
        app = yaml.safe_load(f)
    with open("configs/dq_thresholds.yaml", "r", encoding="utf-8") as f:
        dq = yaml.safe_load(f)
    return app, dq


def _ocr_image(image_path: Path, profile: str) -> str:
    """Placeholder OCR engine returning dummy text."""
    # Real implementation would call tesseract or similar
    return f"ocr({profile})"


def run_ocr(cache_dir: str) -> None:
    app_cfg, dq_cfg = _load_configs()
    pages_path = Path(cache_dir) / "pages.parquet"
    if not pages_path.exists():
        return
    pages = pd.read_parquet(pages_path)
    ocr_rows: List[dict] = []
    text_rows: List[dict] = []

    cer_max = dq_cfg["ocr"]["cer_max"]
    suspect_max = dq_cfg["ocr"]["suspect_char_ratio_max"]

    for _, row in pages.iterrows():
        text_final = row.get("text_raw")
        profile_used = None
        proxy_cer = None
        suspect_ratio = None
        if pd.isna(text_final):
            # Need OCR
            for profile in app_cfg["ocr"]["profiles"].keys():
                text_candidate = _ocr_image(Path(row["image_path"]), profile)
                proxy_cer = 0.0
                suspect_ratio = 0.0
                profile_used = profile
                if proxy_cer <= cer_max and suspect_ratio <= suspect_max:
                    text_final = text_candidate
                    break
            if text_final is None:
                text_final = ""
        ocr_rows.append(
            {
                "doc_name": row.doc_name,
                "page_number": row.page_number,
                "text_ocr": text_final if pd.isna(row.text_raw) else None,
                "profile_used": profile_used,
                "proxy_cer": proxy_cer,
                "suspect_char_ratio": suspect_ratio,
            }
        )
        text_rows.append(
            {
                "doc_name": row.doc_name,
                "page_number": row.page_number,
                "text_final": row.text_raw if pd.notna(row.text_raw) else text_final,
            }
        )

    pd.DataFrame(ocr_rows).to_parquet(Path(cache_dir) / "ocr_text.parquet", index=False)
    pd.DataFrame(text_rows).to_parquet(Path(cache_dir) / "page_text.parquet", index=False)

    pd.DataFrame(ocr_rows).to_csv("logs/ocr_metrics.csv", index=False)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    run_ocr(args.cache)
