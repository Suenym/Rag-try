# src/cli/eval_retrieval_public.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
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


def _load_public_qa(app_cfg: dict) -> List[Dict[str, Any]]:
    """Склеиваем вопросы (xlsx) и ответы (json) в единый список dict-ов."""
    a_path = Path(app_cfg["paths"]["public_answers"])
    q_path = Path(app_cfg["paths"]["public_questions"])

    with open(a_path, "r", encoding="utf-8") as f:
        answers = json.load(f)

    dfa = pd.DataFrame(answers)
    # ответ может быть разного типа -> приведём к строке для matcher-а, а оригинал сохраним
    dfa["answer_raw"] = dfa["answer"]
    dfa["answer"] = dfa["answer"].astype(str)

    dfq = pd.read_excel(q_path)

    # Автоопределение колонки с текстом вопроса
    q_col = None
    for cand in ["question", "query", "full_question", "text", "q"]:
        if cand in dfq.columns:
            q_col = cand
            break
    if not q_col:
        raise ValueError("В файле вопросов нет подходящей колонки с текстом вопроса "
                         "(ожидались: question | query | full_question | text | q)")
    typer.echo(f"ℹ️ Колонка с вопросом: '{q_col}'")

    # Оставим минимум нужного
    dfq = dfq[["question_id", q_col]].rename(columns={q_col: "question"})

    # Джоин по question_id
    df = dfq.merge(dfa[["question_id", "answer", "answer_raw", "relevant_chunks"]],
                   on="question_id", how="inner")

    # В удобный список словарей
    items: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        items.append({
            "question_id": int(row["question_id"]),
            "question": str(row["question"]),
            "answer": row["answer"],          # строковый вид — для поиска по паттернам
            "answer_raw": row["answer_raw"],  # оригинальное значение — на всякий
            "relevant_chunks": row.get("relevant_chunks", []),
        })
    return items


def _search_candidates(
    retr: Retriever,
    query: str,
    k: int,
    reranker: Optional[Reranker],
    rerank_topn: int,
) -> List[Dict[str, Any]]:
    """Достаём кандидатов и (опц.) переранжируем."""
    # если есть reranker — нужно оверфетчить достаточно кандидатов
    fetch_k = max(k, min(max(k * 5, k), rerank_topn) if reranker else k)
    hits = retr.search(query, k=fetch_k, overfetch=fetch_k if reranker else None)
    if reranker:
        hits = reranker.rerank(query, hits, top_k=k)
    else:
        hits = hits[:k]
    return hits


@app.command()
def main(
    index: str = typer.Option("data/index", "--index"),
    cache: str = typer.Option("data/cache", "--cache"),  # зарезервировано под будущие фичи
    k: int = typer.Option(10, "--k"),
    device: Optional[str] = typer.Option(None, "--device", help="cpu | cuda | mps (auto if omitted)"),
    rerank_model: Optional[str] = typer.Option(None, "--rerank-model"),
    rerank_topn: int = typer.Option(50, "--rerank-topn", help="сколько кандидатов переранжировать"),
    dump_topk: Optional[str] = typer.Option(None, "--dump-topk", help="путь к CSV с топ-K результатами"),
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
        answer_str = str(item["answer"])  # для релакс-паттернов
        answer_raw = item.get("answer_raw")

        hits = _search_candidates(retr, query, k, rr, rerank_topn)

        # подготовка паттернов на каждый вопрос (один раз)
        pat_list = relaxed_answer_match.compile_patterns(answer_str)

        # найти ранг первого совпадения
        rank_found: Optional[int] = None
        matched_pat = ""
        for rank, h in enumerate(hits, start=1):
            # проверяем preview (и при желании можно подставить полный текст чанка, если храните)
            ok, pat = relaxed_answer_match.any_match(h["preview"], pat_list)
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

        details_rows.append({
            "qid": qid,
            "top_hit_rank": "MISS" if rank_found is None else rank_found,
            "doc_name": "" if rank_found is None else hits[rank_found-1]["doc_name"],
            "page_number": "" if rank_found is None else hits[rank_found-1]["page_number"],
            "matched_pattern": matched_pat,
            "score": hits[0]["rerank_score"] if (hits and "rerank_score" in hits[0]) else (hits[0]["score"] if hits else 0.0),
        })

        # дампим весь топ-K по каждому вопросу, если попросили
        if dump_topk:
            for rank, h in enumerate(hits, start=1):
                dump_rows.append({
                    "question_id": qid,
                    "question": query,
                    "answer_raw": answer_raw,
                    "rank": rank,
                    "doc_name": h["doc_name"],
                    "page_number": h["page_number"],
                    "score": h.get("score"),
                    "rerank_score": h.get("rerank_score"),
                    "preview": h.get("preview", ""),
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
        lines.append(f"| {r['qid']} | {r['top_hit_rank']} | {r['doc_name']} | {r['page_number']} | {r['matched_pattern']} | {r['score']} |")
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # 5) CSV с топ-K (опционально)
    if dump_topk:
        dump_path = Path(dump_topk)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(dump_rows).to_csv(dump_path, index=False, encoding="utf-8-sig")

    # 6) понятный вывод в консоль
    msg = (f"Saved report to {report_path} • "
           f"R@1 {recall1:.3f} | R@3 {recall3:.3f} | R@5 {recall5:.3f} | R@10 {recall10:.3f} | MRR@10 {mrr10:.3f}")
    if dump_topk:
        msg += f" • Top-K CSV: {dump_topk}"
    typer.echo(msg)


if __name__ == "__main__":
    app()
