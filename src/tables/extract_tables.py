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


def extract_tables(cache_dir: str) -> None:
    app_cfg, dq_cfg = _load_configs()
    text_path = Path(cache_dir) / "page_text.parquet"
    if not text_path.exists():
        return
    pages = pd.read_parquet(text_path)
    table_rows: List[dict] = []
    metrics_rows: List[dict] = []

    struct_min = dq_cfg["tables"]["struct_valid_min"]
    numeric_min = dq_cfg["tables"]["numeric_ratio_min"]

    for _, row in pages.iterrows():
        # Placeholder: no real table extraction; just dummy based on pattern
        mode_used = app_cfg["tables"]["prefer"][0]
        # For demo, treat lines containing '|' as tables
        tables_in_page = [line for line in row.text_final.splitlines() if "|" in line]
        for tid, line in enumerate(tables_in_page, start=1):
            cols = [c.strip() for c in line.split("|")]
            numeric_ratio = sum(c.replace(".", "", 1).isdigit() for c in cols) / max(len(cols), 1)
            struct_valid = 1.0 if len(cols) > 1 else 0.0
            risky = struct_valid < struct_min or numeric_ratio < numeric_min
            table_rows.append(
                {
                    "doc_name": row.doc_name,
                    "page_number": row.page_number,
                    "table_id": tid,
                    "row_idx": 0,
                    "col_count": len(cols),
                    "numeric_ratio": numeric_ratio,
                    "header_text": cols[0],
                    "row_text": " | ".join(cols),
                    "risky_table": risky,
                    "mode_used": mode_used,
                }
            )
            metrics_rows.append(
                {
                    "doc_name": row.doc_name,
                    "page_number": row.page_number,
                    "table_id": tid,
                    "struct_valid": struct_valid,
                    "numeric_ratio": numeric_ratio,
                    "mode_used": mode_used,
                    "risky_table": risky,
                }
            )

    df = pd.DataFrame(table_rows)
    df.to_parquet(Path(cache_dir) / "tables.parquet", index=False)
    pd.DataFrame(metrics_rows).to_csv("logs/tables_metrics.csv", index=False)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    extract_tables(args.cache)
