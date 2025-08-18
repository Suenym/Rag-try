from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
import re

import pandas as pd
import yaml


# ---------------------------
# Config & normalization utils
# ---------------------------

def _load_configs() -> dict:
    with open("configs/app.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _normalize_text(text: str, cfg: dict, log: List[dict], context: dict) -> str:
    """
    Minimal, deterministic normalization:
      - repair hyphenation line-breaks
      - collapse whitespace
    """
    original = text or ""
    s = original

    # Hyphen repair: remove "-\n" (optionally with spaces) when next token starts with a letter (Latin/Cyrillic)
    if cfg.get("normalize", {}).get("hyphen_fix_enabled", True):
        s = re.sub(r"-\s*\n(?=[A-Za-zА-Яа-яЁё])", "", s)

    # Normalize non-breaking spaces and collapse whitespace
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()

    if s != original:
        entry = {"change": "text", "before": original, "after": s}
        entry.update(context or {})
        log.append(entry)
    return s


# ---------------------------
# Main entry
# ---------------------------

def apply_normalization(cache_dir: str) -> None:
    """
    Reads page_text.parquet / tables.parquet and writes:
      - page_text_norm.parquet  (doc_name, page_number, text_norm)
      - tables_norm.parquet     (doc_name, page_number, table_id, row_idx, row_text_norm, header_text_norm)

    Empty-safe: if inputs are missing or 0×0, writes empty files with the correct schema.
    """
    cfg = _load_configs()
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    text_in = cache / "page_text.parquet"
    text_out = cache / "page_text_norm.parquet"

    tables_in = cache / "tables.parquet"
    tables_out = cache / "tables_norm.parquet"

    changes: List[Dict[str, Any]] = []

    # ---------- TEXT ----------
    text_cols_required = ["doc_name", "page_number", "text_final"]
    text_out_cols = ["doc_name", "page_number", "text_norm"]

    if not text_in.exists():
        print(f"[normalize] {text_in} not found -> writing empty {text_out.name}")
        pd.DataFrame(columns=text_out_cols).to_parquet(text_out, index=False)
    else:
        df = pd.read_parquet(text_in)

        # If someone wrote a 0×0 frame (no columns), fix to schema
        if df.empty and len(df.columns) == 0:
            print("[normalize] page_text.parquet is 0×0 -> writing empty normalized with schema")
            pd.DataFrame(columns=text_out_cols).to_parquet(text_out, index=False)
        else:
            # Ensure required columns exist
            for col in text_cols_required:
                if col not in df.columns:
                    df[col] = None

            # Build normalized column
            df["text_norm"] = df["text_final"].fillna("").map(
                lambda s: _normalize_text(
                    s, cfg, changes,
                    {"field": "page", "doc_name": None, "page_number": None}
                )
            )

            # Keep only desired columns and ensure types are writeable
            df_out = df[text_out_cols].copy()
            df_out.to_parquet(text_out, index=False)
            print(f"[normalize] wrote {text_out} rows={len(df_out)}")

    # ---------- TABLES ----------
    table_in_cols_known = ["doc_name", "page_number", "table_id", "row_idx", "row_text", "header_text"]
    table_out_cols = ["doc_name", "page_number", "table_id", "row_idx", "row_text_norm", "header_text_norm"]

    if not tables_in.exists():
        print(f"[normalize] {tables_in} not found -> writing empty {tables_out.name}")
        pd.DataFrame(columns=table_out_cols).to_parquet(tables_out, index=False)
    else:
        tdf = pd.read_parquet(tables_in)

        # If 0×0, write empty schema
        if tdf.empty and len(tdf.columns) == 0:
            print("[normalize] tables.parquet is 0×0 -> writing empty normalized with schema")
            pd.DataFrame(columns=table_out_cols).to_parquet(tables_out, index=False)
        else:
            # Ensure known input columns exist (fill if absent)
            for col in table_in_cols_known:
                if col not in tdf.columns:
                    tdf[col] = None

            # Build normalized columns (both row_text and header_text, even if header_text is None)
            tdf["row_text_norm"] = tdf["row_text"].fillna("").map(
                lambda s: _normalize_text(
                    s, cfg, changes,
                    {"field": "table_row", "doc_name": None, "page_number": None}
                )
            )
            tdf["header_text_norm"] = tdf["header_text"].fillna("").map(
                lambda s: _normalize_text(
                    s, cfg, changes,
                    {"field": "table_header", "doc_name": None, "page_number": None}
                )
            )

            # Compose output with enforced schema
            out = pd.DataFrame({
                "doc_name": tdf.get("doc_name"),
                "page_number": tdf.get("page_number"),
                "table_id": tdf.get("table_id"),
                "row_idx": tdf.get("row_idx"),
                "row_text_norm": tdf.get("row_text_norm"),
                "header_text_norm": tdf.get("header_text_norm"),
            })
            out.to_parquet(tables_out, index=False)
            print(f"[normalize] wrote {tables_out} rows={len(out)}")

    # ---------- LOG ----------
    changes_df = (
        pd.DataFrame(changes)
        if changes
        else pd.DataFrame([], columns=["change", "before", "after", "doc_name", "page_number", "field"])
    )
    changes_df.to_csv(logs_dir / "normalize_changes.csv", index=False)
    print(f"[normalize] log -> {logs_dir / 'normalize_changes.csv'} (rows={len(changes_df)})")


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    apply_normalization(args.cache)
