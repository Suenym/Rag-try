from __future__ import annotations

from pathlib import Path
from typing import List
import re

import pandas as pd
import yaml


def _load_configs() -> dict:
    with open("configs/app.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_text(text: str, cfg: dict, log: List[dict], context: dict) -> str:
    original = text
    if cfg.get("normalize", {}).get("hyphen_fix_enabled", True):
        text = re.sub(r"-\n(?=[a-zа-я])", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text != original:
        log.append({"change": "text", "before": original, "after": text, **context})
    return text


def apply_normalization(cache_dir: str) -> None:
    cfg = _load_configs()
    cache = Path(cache_dir)
    text_path = cache / "page_text.parquet"
    table_path = cache / "tables.parquet"
    changes: List[dict] = []

    if text_path.exists():
        df = pd.read_parquet(text_path)
        df["text_norm"] = df.apply(
            lambda r: _normalize_text(r.text_final or "", cfg, changes, {"doc_name": r.doc_name, "page_number": r.page_number, "field": "page"}),
            axis=1,
        )
        df[["doc_name", "page_number", "text_norm"]].to_parquet(cache / "page_text_norm.parquet", index=False)

    if table_path.exists():
        tdf = pd.read_parquet(table_path)
        tdf["row_text_norm"] = tdf.apply(
            lambda r: _normalize_text(r.row_text or "", cfg, changes, {"doc_name": r.doc_name, "page_number": r.page_number, "field": "table_row"}),
            axis=1,
        )
        tdf["header_text_norm"] = tdf.apply(
            lambda r: _normalize_text(r.header_text or "", cfg, changes, {"doc_name": r.doc_name, "page_number": r.page_number, "field": "table_header"}),
            axis=1,
        )
        tdf.to_parquet(cache / "tables_norm.parquet", index=False)

    if changes:
        pd.DataFrame(changes).to_csv("logs/normalize_changes.csv", index=False)
    else:
        pd.DataFrame([], columns=["change", "before", "after", "doc_name", "page_number", "field"]).to_csv(
            "logs/normalize_changes.csv", index=False
        )


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    apply_normalization(args.cache)
