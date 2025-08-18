from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml


def _load_configs() -> tuple[dict, dict]:
    with open("configs/app.yaml", "r", encoding="utf-8") as f:
        app = yaml.safe_load(f)
    with open("configs/dq_thresholds.yaml", "r", encoding="utf-8") as f:
        dq = yaml.safe_load(f)
    return app, dq


def _variants(answer: str) -> List[str]:
    answer = str(answer)
    variants = {answer.lower()}
    if answer.replace(" ", "").isdigit():
        digits = answer.replace(" ", "")
        variants.add(digits)
        variants.add(f"{int(digits):,}".replace(",", " "))
    return list(variants)


def answer_findability(cache_dir: str) -> Dict[str, float]:
    app_cfg, dq_cfg = _load_configs()
    answers_path = Path(app_cfg["paths"]["public_answers"])
    with open(answers_path, "r", encoding="utf-8") as f:
        answers = json.load(f)
    page_df = pd.read_parquet(Path(cache_dir) / "page_text_norm.parquet")
    table_df = (
        pd.read_parquet(Path(cache_dir) / "tables_norm.parquet")
        if (Path(cache_dir) / "tables_norm.parquet").exists()
        else pd.DataFrame()
    )

    hits: List[dict] = []
    found = 0
    for item in answers:
        qid = item["question_id"]
        ans = item["answer"]
        variants = _variants(ans)
        hit = False
        for _, row in page_df.iterrows():
            text = str(row.text_norm).lower()
            if any(v in text for v in variants):
                hits.append({"question_id": qid, "doc": row.doc_name, "page": row.page_number, "match_type": "page"})
                hit = True
                break
        if not hit and not table_df.empty:
            for _, row in table_df.iterrows():
                text = str(row.row_text_norm).lower()
                if any(v in text for v in variants):
                    hits.append({"question_id": qid, "doc": row.doc_name, "page": row.page_number, "match_type": "table"})
                    hit = True
                    break
        if hit:
            found += 1

    coverage = found / max(len(answers), 1)
    pd.DataFrame(hits).to_csv("logs/public_findability_hits.csv", index=False)
    with open("reports/public_findability.md", "w", encoding="utf-8") as f:
        f.write(f"Coverage: {coverage:.2%}\n")
        for h in hits[:10]:
            f.write(f"- q{h['question_id']} {h['doc']} p{h['page']} ({h['match_type']})\n")
    return {"coverage": coverage, "target": dq_cfg["public"]["answer_findability_target"]}


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    answer_findability(args.cache)
