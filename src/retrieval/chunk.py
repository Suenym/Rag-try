from __future__ import annotations

from pathlib import Path
from hashlib import md5
from typing import List

import pandas as pd


def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks by character count."""
    text = text or ""
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks


def build_chunks(
    cache_dir: str,
    out_dir: str,
    page_chunk_chars: int = 1200,
    overlap_chars: int = 200,
    min_chars: int = 80,
) -> pd.DataFrame:
    """Build chunk dataframe from normalized page and table text."""
    cache = Path(cache_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    chunk_id = 0

    page_path = cache / "page_text_norm.parquet"
    if page_path.exists():
        df_pages = pd.read_parquet(page_path)
        df_pages = df_pages.sort_values(["doc_name", "page_number"])
        for _, row in df_pages.iterrows():
            text = row.get("text_norm") or ""
            for part in _split_text(text, page_chunk_chars, overlap_chars):
                if len(part) < min_chars:
                    continue
                rows.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_name": row["doc_name"],
                        "page_number": int(row["page_number"]),
                        "source_type": "page",
                        "text": part,
                        "text_len": len(part),
                        "token_est": len(part) // 4,
                        "hash": md5(part.encode("utf-8")).hexdigest(),
                    }
                )
                chunk_id += 1

    table_path = cache / "tables_norm.parquet"
    if table_path.exists():
        df_tab = pd.read_parquet(table_path)
        df_tab = df_tab.sort_values(["doc_name", "page_number"])  # type: ignore

        # header chunks (unique per table)
        hdr_cols = ["doc_name", "page_number", "header_text_norm"]
        headers = (
            df_tab[hdr_cols]
            .dropna(subset=["header_text_norm"])
            .drop_duplicates()
            .itertuples(index=False)
        )
        for h in headers:
            text = str(h.header_text_norm)
            if len(text) < min_chars:
                continue
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "doc_name": h.doc_name,
                    "page_number": int(h.page_number),
                    "source_type": "table_header",
                    "text": text,
                    "text_len": len(text),
                    "token_est": len(text) // 4,
                    "hash": md5(text.encode("utf-8")).hexdigest(),
                }
            )
            chunk_id += 1

        # row chunks
        for _, row in df_tab.iterrows():
            header = row.get("header_text_norm") or ""
            row_text = row.get("row_text_norm") or ""
            text = f"{header} | {row_text}" if header else str(row_text)
            if len(text) < min_chars:
                continue
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "doc_name": row["doc_name"],
                    "page_number": int(row["page_number"]),
                    "source_type": "table_row",
                    "text": text,
                    "text_len": len(text),
                    "token_est": len(text) // 4,
                    "hash": md5(text.encode("utf-8")).hexdigest(),
                }
            )
            chunk_id += 1

    df_chunks = pd.DataFrame(rows)
    df_chunks.to_parquet(out / "chunks.parquet", index=False)
    return df_chunks

