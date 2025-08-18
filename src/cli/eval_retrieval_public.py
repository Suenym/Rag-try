import json
from pathlib import Path
from typing import List

import pandas as pd
import typer

from src.retrieval.search import SearchIndex
from src.validate.match_utils import match_answer

app = typer.Typer(add_completion=False)

CANDIDATE_ID_COLS = ["question_id", "id", "qid", "QuestionID", "questionId"]


def pick_id_col(cols: List[str]):
    colmap = {c.lower(): c for c in cols}
    for cand in CANDIDATE_ID_COLS:
        if cand.lower() in colmap:
            return colmap[cand.lower()]
    return None


@app.command()
def main(
    index: str = "data/index",
    cache: str = "data/cache",  # not used but kept for interface parity
    k: int = 10,
    questions_path: str = "data/public/questions_public.xlsx",
    answers_path: str = "data/public/answers_public.json",
    report_path: str = "reports/retrieval_public.md",
    diag_path: str = "data/diagnostics/retrieval_public.csv",
):
    searcher = SearchIndex(index)

    df_q = pd.read_excel(questions_path)
    id_col = pick_id_col(df_q.columns)
    if id_col and id_col != "question_id":
        df_q = df_q.rename(columns={id_col: "question_id"})

    answers = json.loads(Path(answers_path).read_text(encoding="utf-8"))
    ans_map = {int(r["question_id"]): r["answer"] for r in answers}

    results = []
    ranks = []

    for row in df_q.itertuples():
        qid = int(row.question_id)
        qtext = getattr(row, "question", "")
        ans = ans_map.get(qid)
        hits = searcher.search(qtext, k=k)
        top_rank = None
        match_doc = match_page = match_pat = None
        for rank, hit in enumerate(hits.itertuples(), start=1):
            pat = match_answer(hit.text, ans)
            if pat:
                top_rank = rank
                match_doc = hit.doc_name
                match_page = hit.page_number
                match_pat = pat
                break
        ranks.append(top_rank)
        results.append(
            {
                "qid": qid,
                "question": qtext,
                "top_hit_rank": top_rank,
                "doc_name": match_doc,
                "page_number": match_page,
                "matched_pattern": match_pat,
                "score": hits.iloc[0]["score"] if not hits.empty else None,
            }
        )

    def _recall_at(rk_list, k):
        return sum(1 for r in rk_list if r and r <= k) / len(rk_list)

    def _mrr_at(rk_list, k):
        return sum(1 / r for r in rk_list if r and r <= k) / len(rk_list)

    recall1 = _recall_at(ranks, 1)
    recall3 = _recall_at(ranks, 3)
    recall5 = _recall_at(ranks, 5)
    recall10 = _recall_at(ranks, 10)
    mrr10 = _mrr_at(ranks, 10)

    diag_df = pd.DataFrame(results)
    diag_path = Path(diag_path)
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    diag_df.to_csv(diag_path, index=False)

    lines = ["# Retrieval evaluation (public questions)", ""]
    lines.append(f"Recall@1: {recall1:.3f}")
    lines.append(f"Recall@3: {recall3:.3f}")
    lines.append(f"Recall@5: {recall5:.3f}")
    lines.append(f"Recall@10: {recall10:.3f}")
    lines.append(f"MRR@10: {mrr10:.3f}")
    lines.append("")
    lines.append("| qid | top_hit_rank | doc_name | page_number | matched_pattern | score |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['qid']} | {r['top_hit_rank'] if r['top_hit_rank'] is not None else 'MISS'} | "
            f"{r['doc_name'] or ''} | {r['page_number'] or ''} | {r['matched_pattern'] or ''} | "
            f"{r['score'] if r['score'] is not None else ''} |"
        )
    Path(report_path).write_text("\n".join(lines), encoding="utf-8")
    typer.secho(
        f"Recall@10={recall10:.3f} | MRR@10={mrr10:.3f} (results saved)",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()

