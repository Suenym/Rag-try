from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import pandas as pd
import typer

from src.retrieval.search import Retriever
from src.retrieval.rerank import Reranker
from src.reader.extract import extract_answer_from_hits
from src.validate.linkcheck import linkcheck

app = typer.Typer(add_completion=False)


def _pick_q_col(df: pd.DataFrame) -> str:
    for c in ["question", "full_question", "query"]:
        if c in df.columns:
            return c
    raise ValueError("Не найдена колонка с текстом вопроса (question|full_question|query)")


def _ensure_qid(df: pd.DataFrame) -> pd.DataFrame:
    if "question_id" in df.columns:
        return df[["question_id"] + [c for c in df.columns if c != "question_id"]]
    if "id" in df.columns:
        df = df.rename(columns={"id": "question_id"})
        return df
    raise ValueError("Нет колонки question_id (или id)")


@app.command()
def main(
    index: str = typer.Option("data/index_e5", "--index"),
    questions_xlsx: str = typer.Option("data/public/questions_public.xlsx", "--questions"),
    out_submission: str = typer.Option("submission/submission.json", "--out"),
    out_md: str = typer.Option("predictions_public.md", "--out-md"),
    k: int = typer.Option(10, "--k"),
    device: Optional[str] = typer.Option(None, "--device"),
    rerank_model: Optional[str] = typer.Option("jinaai/jina-reranker-v2-base-multilingual", "--rerank-model"),
    rerank_topn: int = typer.Option(300, "--rerank-topn"),
    hybrid: bool = typer.Option(True, "--hybrid"),
):
    Path(out_submission).parent.mkdir(parents=True, exist_ok=True)

    dfq = pd.read_excel(questions_xlsx)
    dfq = _ensure_qid(dfq)
    qcol = _pick_q_col(dfq)
    dfq = dfq[["question_id", qcol]].rename(columns={qcol: "question"})

    retr = Retriever.from_dir(Path(index), device=device)
    rr = Reranker(rerank_model, device=device) if rerank_model else None

    preds_rows = []
    sub_out: List[Dict[str, Any]] = []

    for row in dfq.to_dict(orient="records"):
        qid = int(row["question_id"])
        q = str(row["question"])

        overfetch = max(k * 5, rerank_topn)
        hits = (
            retr.search_hybrid(q, k=overfetch, overfetch=overfetch)
            if hybrid
            else retr.search(q, k=overfetch, overfetch=overfetch)
        )
        if rr:
            hits = rr.rerank(q, hits, top_k=overfetch)
        hits = hits[:k]

        ans = extract_answer_from_hits(q, hits, topk=min(5, len(hits)))
        best_idx = linkcheck(ans.value, ans.atype, hits, topk=min(5, len(hits)))

        rel: List[Dict[str, Any]] = []
        if best_idx >= 0:
            h = hits[best_idx]
            rel.append(
                {
                    "document_name": h.get("doc_name", ""),
                    "page_number": int(h.get("page_number", -1) or -1),
                }
            )
        else:
            if hits:
                h = hits[0]
                rel.append(
                    {
                        "document_name": h.get("doc_name", ""),
                        "page_number": int(h.get("page_number", -1) or -1),
                    }
                )

        preds_rows.append(
            {
                "qid": qid,
                "question": q,
                "prediction": ans.value,
                "source": f"{rel[0]['document_name']}:{rel[0]['page_number']}" if rel else "",
                "link_ok": "✅" if best_idx >= 0 and ans.value != "N/A" else "❌",
            }
        )

        sub_out.append(
            {
                "question_id": qid,
                "relevant_chunks": rel if rel else [],
                "answer": ans.value,
            }
        )

    lines = [
        "# Public QA — predictions",
        "",
        "| qid | вопрос | предсказание | источник | link |",
        "|---:|---|---|---|:---:|",
    ]
    for r in preds_rows:
        lines.append(
            f"| {r['qid']} | {r['question']} | {r['prediction']} | {r['source']} | {r['link_ok']} |"
        )
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")

    Path(out_submission).write_text(
        json.dumps(sub_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    typer.echo(f"Saved: {out_md} and {out_submission}")


if __name__ == "__main__":
    app()
