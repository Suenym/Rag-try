# src/cli/eval_retrieval_public.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import re
import pandas as pd
import typer

from src.retrieval.search import Retriever
from src.retrieval.rerank import Reranker
from src.retrieval.llm_rerank import GeminiReranker
from src.validate.match_utils import relaxed_answer_match


app = typer.Typer(add_completion=False)


def _load_app_cfg() -> dict:
    import yaml
    with open("configs/app.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _light_answer_norm(s: str) -> str:
    """Лёгкая нормализация ответа для построения паттернов."""
    if s is None:
        return ""
    s = str(s)
    # возможные следы «моджибейка» кавычек
    s = (
        s.replace("«", '"').replace("»", '"')
         .replace("Â«", '"').replace("Â»", '"')
    )
    # NBSP → пробел
    s = s.replace("\u00A0", " ")
    # схлопываем пробелы
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
    dfa["answer_raw"] = dfa["answer"]          # сохраним исходный тип/значение
    dfa["answer"] = dfa["answer"].map(_light_answer_norm)

    dfq = pd.read_excel(q_path)

    # Автоопределение колонки с текстом вопроса
    q_col = None
    for cand in ["question", "query", "full_question"]:
        if cand in dfq.columns:
            q_col = cand
            break
    if not q_col:
        raise ValueError("В файле вопросов нет колонки с текстом вопроса (question | query | full_question)")
    typer.echo(f"ℹ️ Колонка с вопросом: '{q_col}'")

    # Автоопределение id-колонки
    id_col = None
    for cand in ["question_id", "id", "qid"]:
        if cand in dfq.columns:
            id_col = cand
            break
    if not id_col:
        raise ValueError("В файле вопросов нет id-колонки (question_id | id | qid)")
    if id_col != "question_id":
        typer.echo(f"ℹ️ Колонка с id: '{id_col}' → переименую в 'question_id'")

    # Оставляем только нужные столбцы и приводим к ожидаемым именам
    dfq = dfq[[id_col, q_col]].rename(columns={id_col: "question_id", q_col: "question"})

    # join с ответами по question_id
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
    llm_reranker: Optional[GeminiReranker] = None,
) -> List[Dict[str, Any]]:
    """Достаём кандидатов и (опц.) переранжируем."""
    need_overfetch = (reranker is not None) or (llm_reranker is not None)
    overfetch = max(5 * k, rerank_topn, k) if need_overfetch else k

    if hybrid:
        hits = retr.search_hybrid(query, k=k, overfetch=overfetch)
    else:
        hits = retr.search(query, k=k, overfetch=overfetch)

    if reranker:
        # сначала HF-реранк на широком пуле
        hits = reranker.rerank(query, hits, top_k=overfetch)

    if llm_reranker:
        # затем — LLM на урезанном пуле и мягкое смешивание внутри самого реранкера
        llm_topn = min(len(hits), max(5 * k, rerank_topn))
        hits = llm_reranker.rerank(query, hits[:llm_topn], top_k=k)
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
    llm_rerank: bool = typer.Option(False, "--llm-rerank", help="доп. LLM-реранк (Gemini)"),
    llm_model: str = typer.Option("gemini-2.5-flash", "--llm-model", help="модель Gemini"),
    llm_alpha: float = typer.Option(0.25, "--llm-alpha", min=0.0, max=1.0, help="вес LLM в смешивании (0..1)"),
    llm_max_chars: int = typer.Option(600, "--llm-max-chars", help="обрезка превью для LLM, символов"),
):
    # 1) данные
    app_cfg = _load_app_cfg()
    qa = _load_public_qa(app_cfg)

    # 2) модели
    retr = Retriever.from_dir(Path(index), device=device)
    rr = Reranker(rerank_model, device=device) if rerank_model else None
    llm_rr = GeminiReranker(model=llm_model, max_chars=llm_max_chars, alpha=llm_alpha) if llm_rerank else None

    # 3) метрики + подробности
    top1 = top3 = top5 = top10 = 0
    mrr10_sum = 0.0
    details_rows: List[Dict[str, Any]] = []
    dump_rows: List[Dict[str, Any]] = []
    patterns_by_qid: Dict[int, List[str]] = {}

    for item in qa:
        qid = int(item["question_id"])
        query = str(item["question"])
        answer_str = str(item["answer"])
        answer_raw = item.get("answer_raw")

        hits = _search_candidates(
            retr,
            query,
            k,
            rr,
            rerank_topn,
            hybrid=hybrid,
            llm_reranker=llm_rr,
        )

        # подготовка паттернов
        pat_list = relaxed_answer_match.compile_patterns(answer_str)
        patterns_by_qid[qid] = pat_list

        # поиск ранга первого совпадения
        rank_found: Optional[int] = None
        matched_pat = ""
        for rank, h in enumerate(hits, start=1):
            text = h.get("text") or h.get("preview", "")
            ok, pat = relaxed_answer_match.any_match(text, pat_list)

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

        # строка для подробного отчёта (только первое попадание)
        if rank_found is not None:
            hit = hits[rank_found - 1]
            doc_name = hit.get("doc_name", "")
            page_number = hit.get("page_number", "")
            hit_score = hit.get("rerank_score", hit.get("score", ""))
        else:
            doc_name = ""
            page_number = ""
            hit_score = ""

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
                ok_rank, pat_rank = relaxed_answer_match.any_match(h.get("preview", ""), pat_list)
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
                    "matched_pattern": pat_rank if ok_rank else "",
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

    # если по какой-то причине details_rows пуст — восстановим из dump_rows
    if not details_rows and dump_rows:
        ddf = pd.DataFrame(dump_rows)
        for qid in sorted(ddf["question_id"].unique()):
            sub = ddf[ddf["question_id"] == qid].sort_values("rank")
            # ищем первую строку с матчем
            m = sub[sub["matched_pattern"] != ""]
            if not m.empty:
                r = m.iloc[0]
                details_rows.append({
                    "qid": int(qid),
                    "top_hit_rank": int(r["rank"]),
                    "doc_name": r.get("doc_name", ""),
                    "page_number": int(r.get("page_number", -1)) if str(r.get("page_number", "")).strip() != "" else "",
                    "matched_pattern": r.get("matched_pattern", ""),
                    "score": r.get("rerank_score", r.get("score", "")),
                })
            elif not sub.empty:
                # иначе берём лучший по rerank_score/score
                sub = sub.copy()
                sub["best_score"] = sub["rerank_score"].fillna(sub["score"])
                r = sub.sort_values("best_score", ascending=False).iloc[0]
                details_rows.append({
                    "qid": int(qid),
                    "top_hit_rank": int(r["rank"]),
                    "doc_name": r.get("doc_name", ""),
                    "page_number": int(r["page_number"]) if str(r.get("page_number", "")).strip() != "" else "",
                    "matched_pattern": "",
                    "score": r.get("rerank_score", r.get("score", "")),
                })

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

    for r in sorted(details_rows, key=lambda x: x["qid"]):
        lines.append(
            f"| {r['qid']} | {r['top_hit_rank']} | {r['doc_name']} | {r['page_number']} | {r['matched_pattern']} | {r['score']} |"
        )

    # Auto error-analysis в отчёт (если метрика низкая)
    miss_lines: List[str] = []
    if recall1 < 0.33 and dump_topk and dump_rows:
        df = pd.DataFrame(dump_rows)
        miss_lines.append("")
        miss_lines.append("## Error analysis (top 5 MISS)")
        miss_lines.append("| qid | best_doc | page | best_score | pattern | preview | reason |")
        miss_lines.append("|---|---|---|---|---|---|---|")
        rows_collected = 0
        for qid in sorted(df["question_id"].unique()):
            sub = df[df["question_id"] == qid].sort_values("rank")
            if (sub["matched_pattern"] == "").all():
                sub = sub.copy()
                sub["best_score"] = sub["rerank_score"].fillna(sub["score"])
                row = sub.sort_values("best_score", ascending=False).iloc[0]
                pattern = patterns_by_qid.get(int(qid), [""])[0] if patterns_by_qid.get(int(qid)) else ""
                preview = str(row.get("preview", "")).replace("\n", " ")[:100]
                miss_lines.append(
                    f"| {int(qid)} | {row.get('doc_name','')} | {int(row.get('page_number',-1))} | {float(row.get('best_score',0) or 0):.3f} | {pattern} | {preview} | no match |"
                )
                rows_collected += 1
                if rows_collected >= 5:
                    break

    report_path.write_text("\n".join(lines + miss_lines), encoding="utf-8")

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


if __name__ == "__main__":  # pragma: no cover
    app()
