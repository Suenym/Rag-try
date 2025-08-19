# src/cli/eval_retrieval_public.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import re
import pandas as pd
import typer

from src.retrieval.search import Retriever
from src.retrieval.rerank import Reranker
from src.validate.match_utils import relaxed_answer_match

app = typer.Typer(add_completion=False)

def _load_app_cfg() -> dict:
    import yaml
    with open("configs/app.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _light_answer_norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("«", '"').replace("»", '"')
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _load_public_qa(app_cfg: dict) -> List[Dict[str, Any]]:
    """Склеиваем вопросы (xlsx) и ответы (json) в единый список dict-ов."""
    a_path = Path(app_cfg["paths"]["public_answers"])
    q_path = Path(app_cfg["paths"]["public_questions"])

    # answers: учитываем BOM
    with open(a_path, "r", encoding="utf-8-sig") as f:
        answers = json.load(f)

    dfa = pd.DataFrame(answers)
    # исходный тип ответа сохраним отдельно
    dfa["answer_raw"] = dfa["answer"]
    dfa["answer"] = dfa["answer"].map(_light_answer_norm)

    dfq = pd.read_excel(q_path)

    # Автоопределение колонки с текстом вопроса
    q_col = None
    for cand in ["question", "query", "full_question"]:
        if cand in dfq.columns:
            q_col = cand
            break
    if not q_col:
        raise ValueError("В файле вопросов нет подходящей колонки с текстом вопроса (question | query | full_question)")
    typer.echo(f"ℹ️ Колонка с вопросом: '{q_col}'")

    dfq = dfq[["question_id", q_col]].rename(columns={q_col: "question"})

    # join
    df = dfq.merge(
        dfa[["question_id", "answer", "answer_raw", "relevant_chunks"]],
        on="question_id",
        how="inner",
    )

    items: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        items.append({
            "question_id": int(row["question_id"]),
            "question": str(row["question"]),
            "answer": row["answer"],          # строка — для паттернов
            "answer_raw": row["answer_raw"],  # оригинал — для CSV
            "relevant_chunks": row.get("relevant_chunks", []),
        })
    return items

def _search_candidates(
    retr: Retriever,
    query: str,
    k: int,
    reranker: Optional[Reranker],
    rerank_topn: int,
    hybrid: bool = False,
) -> List[Dict[str, Any]]:
    """Достаём кандидатов и (опц.) переранжируем."""
    overfetch = max(5 * k, rerank_topn, k) if reranker else k

    if hybrid:
        hits = retr.search_hybrid(query, k=k, overfetch=overfetch)
    else:
        hits = retr.search(query, k=k, overfetch=overfetch)

    if reranker:
        hits = reranker.rerank(query, hits, top_k=k)
    else:
        hits = hits[:k]
    return hits

@app.command()
def main(
    index: str = typer.Option("data/index", "--index"),
    cache: str = typer.Option("data/cache", "--cache"),  # резерв под будущие фичи
    k: int = typer.Option(10, "--k"),
    device: Optional[str] = typer.Option(None, "--device", help="cpu | cuda | mps (auto if omitted)"),
    rerank_model: Optional[str] = typer.Option(None, "--rerank-model"),
    rerank_topn: int = typer.Option(50, "--rerank-topn", help="сколько кандидатов переранжировать"),
    dump_topk: Optional[str] = typer.Option(None, "--dump-topk", help="путь к CSV с топ-K результатами"),
    hybrid: bool = typer.Option(False, "--hybrid", help="гибридный BM25+dense поиск"),
):
    # 1) данные
    app_cfg = _load_app_cfg()
    qa = _load_public_qa(app_cfg)

    # 2) модели
    retr = Retriever.from_dir(Path(index), device=device)
    rr = Reranker(rerank_model, device=device) if rerank_model else None

    # 3) метрики + подробности
    top1 = top3 = top5 = top10 = 0
    mrr10_sum = 0.0
    details_rows: List[Dict[str, Any]] = []
    dump_rows: List[Dict[str, Any]] = []

    for item in qa:
        qid = int(item["question_id"])
        query = str(item["question"])
        answer_str = str(item["answer"])
        answer_raw = item.get("answer_raw")

        hits = _search_candidates(retr, query, k, rr, rerank_topn, hybrid=hybrid)

        # подготовка паттернов
        pat_list = relaxed_answer_match.compile_patterns(answer_str)

        # поиск ранга первого совпадения
        rank_found: Optional[int] = None
        matched_pat = ""
        for rank, h in enumerate(hits, start=1):
            ok, pat = relaxed_answer_match.any_match(h.get("preview", ""), pat_list)
            if ok:
                rank_found = rank
                matched_pat = pat
                break

        # метрики
        if rank_found is not None:
            if rank_found == 1:
                top1 += 1
                mrr10_sum += 1.0
            elif rank_found <= 3:
                top3 += 1
                mrr10_sum += 1.0 / rank_found
            elif rank_found <= 5:
                top5 += 1
                mrr10_sum += 1.0 / rank_found
            elif rank_found <= 10:
                top10 += 1
                mrr10_sum += 1.0 / rank_found

        # запись в подробности отчёта
        if rank_found is not None:
            hit = hits[rank_found - 1]
            doc_name = hit.get("doc_name", "")
            page_number = hit.get("page_number", "")
            hit_score = hit.get("rerank_score", hit.get("score", ""))
        else:
            doc_name = ""
            page_number = ""
            hit_score = hits[0].get("rerank_score", hits[0].get("score", 0.0)) if hits else 0.0

        details_rows.append({
            "qid": qid,
            "top_hit_rank": "MISS" if rank_found is None else rank_found,
            "doc_name": doc_name,
            "page_number": page_number,
            "matched_pattern": matched_pat,
            "score": hit_score,
        })

        # CSV dump top-K
        if dump_topk:
            for rank, h in enumerate(hits, start=1):
                dump_rows.append({
                    "question_id": qid,
                    "question": query,
                    "answer_raw": answer_raw,
                    "answer_type": type(answer_raw).__name__ if answer_raw is not None else "",
                    "rank": rank,
                    "score": h.get("score"),
                    "rerank_score": h.get("rerank_score"),
                    "doc_name": h.get("doc_name", ""),
                    "page_number": h.get("page_number", ""),
                    "preview": h.get("preview", ""),
                    "matched_pattern": matched_pat if (rank_found == rank) else "",
                })

    n = max(len(qa), 1)
    recall1 = top1 / n
    recall3 = (top1 + top3) / n
    recall5 = (top1 + top3 + top5) / n
    recall10 = (top1 + top3 + top5 + top10) / n
    mrr10 = mrr10_sum / n

    # 4) отчёт в Markdown
    report_path = Path("reports/retrieval_public.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Retrieval evaluation (public questions)",
        "",
        f"Recall@1: {recall1:.3f}",
        f"Recall@3: {recall3:.3f}",
        f"Recall@5: {recall5:.3f}",
        f"Recall@10: {recall10:.3f}",
        f"MRR@10: {mrr10:.3f}",
        "",
        "| qid | top_hit_rank | doc_name | page_number | matched_pattern | score |",
        "|---|---|---|---|---|---|",
    ]
    for r in details_rows:
        lines.append(
            f"| {r['qid']} | {r['top_hit_rank']} | {r['doc_name']} | {r['page_number']} | {r['matched_pattern']} | {r['score']} |"
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # CSV
    csv_path_str = ""
    if dump_topk:
        csv_path = Path(dump_topk)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(dump_rows).to_csv(csv_path, index=False, encoding="utf-8")
        csv_path_str = f" • Top-K CSV: {csv_path.as_posix()}"

    typer.echo(
        f"Saved report to {report_path.as_posix()} • R@1 {recall1:.3f} | R@3 {recall3:.3f} | R@5 {recall5:.3f} | R@10 {recall10:.3f} | MRR@10 {mrr10:.3f}{csv_path_str}"
    )

    # Простой auto error-analysis, если метрика низкая
    if recall1 < 0.33 and dump_topk:
        df = pd.DataFrame(dump_rows)
        misses = []
        for qid in sorted(df["question_id"].unique()):
            sub = df[df["question_id"] == qid].sort_values("rank")
            if (sub["matched_pattern"] == "").all():
                # лучший кандидат по rerank_score/score
                sub = sub.copy()
                sub["best_score"] = sub["rerank_score"].fillna(sub["score"])
                row = sub.sort_values("best_score", ascending=False).iloc[0]
                misses.append((qid, row.get("doc_name", ""), int(row.get("page_number", -1))))
        if misses:
            typer.echo("Error analysis (top 5 MISS):")
            for qid, dn, pn in misses[:5]:
                typer.echo(f"  - qid={qid}: best candidate {dn}:{pn}")
