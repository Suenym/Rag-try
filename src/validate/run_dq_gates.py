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


def run_dq_gates(cache_dir: str) -> None:
    app_cfg, dq_cfg = _load_configs()
    cache = Path(cache_dir)
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # OCR
    ocr_path = cache / "ocr_text.parquet"
    if ocr_path.exists():
        ocr = pd.read_parquet(ocr_path)
        passed = ocr[
            (ocr["proxy_cer"].fillna(1.0) <= dq_cfg["ocr"]["cer_max"])
            & (ocr["suspect_char_ratio"].fillna(1.0) <= dq_cfg["ocr"]["suspect_char_ratio_max"])
        ]
        ratio = len(passed) / max(len(ocr), 1)
        fail_pages = ocr.loc[~ocr.index.isin(passed.index), ["doc_name", "page_number"]]
        with open(reports_dir / "dq_ocr_report.md", "w", encoding="utf-8") as f:
            f.write("# DQ OCR Report\n\n")
            f.write(f"Успешных страниц: {len(passed)} из {len(ocr)} ({ratio:.2%})\n\n")
            if not fail_pages.empty:
                f.write("## Problem pages\n")
                for _, r in fail_pages.iterrows():
                    f.write(f"- {r.doc_name} p{r.page_number}\n")
    # Tables
    tables_path = cache / "tables.parquet"
    if tables_path.exists() and len(pd.read_parquet(tables_path)):
        tables = pd.read_parquet(tables_path)
        ok = tables[
            (tables["risky_table"] == False)
            & (tables["numeric_ratio"] >= dq_cfg["tables"]["numeric_ratio_min"])
        ]
        ratio = len(ok) / max(len(tables), 1)
        risky = tables[tables["risky_table"]]
        with open(reports_dir / "dq_tables_report.md", "w", encoding="utf-8") as f:
            f.write("# DQ Tables Report\n\n")
            f.write(f"Допустимых таблиц: {len(ok)} из {len(tables)} ({ratio:.2%})\n\n")
            if not risky.empty:
                f.write("## Risky tables\n")
                for _, r in risky.iterrows():
                    f.write(f"- {r.doc_name} p{r.page_number} t{r.table_id}\n")
    # Normalization summary
    norm_log = Path("logs/normalize_changes.csv")
    changes = pd.read_csv(norm_log) if norm_log.exists() else pd.DataFrame()
    with open(reports_dir / "dq_norm_report.md", "w", encoding="utf-8") as f:
        f.write("# DQ Normalization Report\n\n")
        f.write(f"Всего изменений: {len(changes)}\n")

    # Summary
    with open(reports_dir / "dq_summary.md", "w", encoding="utf-8") as f:
        f.write("# DQ Summary\n")
        f.write(f"OCR pages: {len(pd.read_parquet(ocr_path)) if ocr_path.exists() else 0}\n")
        f.write(f"Tables: {len(pd.read_parquet(tables_path)) if tables_path.exists() else 0}\n")
        f.write(f"Normalization changes: {len(changes)}\n")


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    run_dq_gates(args.cache)
