# src/cli/predict_public.py
from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import typer

from src.retrieval.search import Retriever
from src.retrieval.rerank import Reranker
from src.retrieval.llm_rerank import GeminiReranker
from src.validate.match_utils import relaxed_answer_match

app = typer.Typer(add_completion=False)

# -------------------- utils --------------------

def _load_app_cfg() -> dict:
    import yaml
    with open("configs/app.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _light_norm(s: Any) -> str:
    if s is None:
        return ""
    s = str(s)
    s = (s.replace("Â«", '"').replace("Â»", '"')
           .replace("«", '"').replace("»", '"')
           .replace("\u00A0", " "))
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _load_public(app_cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает (dfq, dfa). dfq: вопросы, dfa: ответы-эталон."""
    q_path = Path(app_cfg["paths"]["public_questions"])
    a_path = Path(app_cfg["paths"]["public_answers"])

    dfq = pd.read_excel(q_path)
    # поддержим оба варианта: question_id или id
    if "question_id" not in dfq.columns and "id" in dfq.columns:
        dfq = dfq.rename(columns={"id": "question_id"})
    # автоопределим колонку текста вопроса
    q_col = None
    for cand in ["question", "query", "full_question"]:
        if cand in dfq.columns:
            q_col = cand
            break
    if not q_col:
        raise ValueError("В файле вопросов не найден столбец (question|query|full_question)")
    typer.echo(f"ℹ️ Колонка с вопросом: '{q_col}'")

    keep_cols = ["question_id", q_col]
    if "answer_type" in dfq.columns:
        keep_cols.append("answer_type")
    dfq = dfq[keep_cols].rename(columns={q_col: "question"})

    with open(a_path, "r", encoding="utf-8-sig") as f:
        answers = json.load(f)
    dfa = pd.DataFrame(answers)
    dfa["answer_raw"] = dfa["answer"]
    dfa["answer"] = dfa["answer"].map(_light_norm)

    return dfq, dfa

def _coerce_answer(ans_text: str, answer_type: str) -> Any:
    """Пытаемся привести ответ к типу из questions_public.xlsx (int|float|str)."""
    t = (answer_type or "").strip().lower()
    s = _light_norm(ans_text)

    if t in ("int", "integer"):
        m = re.search(r"-?\d[\d\s\u00A0\u2009]*", s)
        if m:
            num = re.sub(r"[^\d\-]", "", m.group())
            try:
                return int(num)
            except Exception:
                pass
        return s

    if t in ("float", "double", "number"):
        m = re.search(r"-?\d[\d\s\u00A0\u2009]*([.,]\d+)?", s)
        if m:
            raw = m.group().replace(" ", "").replace("\u00A0", "").replace("\u2009", "")
            raw = raw.replace(",", ".")
            try:
                return float(raw)
            except Exception:
                pass
        return s

    # строка
    s = s.strip().strip('"').strip("'")
    return s

def _build_context_snippets(hits: List[Dict[str, Any]], max_hits: int = 6, max_chars: int = 800) -> str:
    parts = []
    for i, h in enumerate(hits[:max_hits], start=1):
        doc = h.get("doc_name", "")
        pg = h.get("page_number", "")
        prev = _light_norm(h.get("preview", ""))[:max_chars]
        parts.append(f"[{i}] {doc}:{pg}\n{prev}")
    return "\n\n".join(parts)

def _choose_relevant_chunks(hits: List[Dict[str, Any]], max_chunks: int = 2) -> List[Dict[str, Any]]:
    """Выбираем top-N ссылок на документы/страницы без дублей."""
    chunks: List[Dict[str, Any]] = []
    seen = set()
    for h in hits:
        doc = h.get("doc_name")
        pg = h.get("page_number")
        if not doc or pg is None or doc == "":
            continue
        try:
            ipg = int(pg)
        except Exception:
            continue
        key = (doc, ipg)
        if key in seen:
            continue
        chunks.append({"document_name": doc, "page_number": ipg})
        seen.add(key)
        if len(chunks) >= max_chunks:
            break
    return chunks

# -------------------- LLM extraction --------------------

def _answer_from_context_gemini(
    question: str,
    contexts: str,
    answer_type: str,
    model_name: str = "gemini-2.5-flash",
) -> str:
    """
    Просим Gemini извлечь короткий ответ ИСКЛЮЧИТЕЛЬНО из данного контекста.
    Если нет ответа — возвращаем 'N/A'.
    """
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    genai.configure(api_key=api_key)

    sys_rules = (
        "Ты — извлекатель ответа из текста. Отвечай КОРОТКО и строго по контексту.\n"
        "Если ответа в контексте нет — ответь 'N/A'.\n"
        "Тип ответа: {typ}. Если тип число — верни только число без слов. Если строка — только сама строка."
    ).format(typ=answer_type or "str")

    prompt = (
        f"{sys_rules}\n\n"
        f"Вопрос: {question}\n\n"
        f"Контекст:\n{contexts}\n\n"
        "Ответ:"
    )

    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(prompt)
    text = (getattr(resp, "text", "") or "").strip()
    text = re.sub(r"^Ответ:\s*", "", text, flags=re.I).strip()
    return text or "N/A"

# -------------------- retrieval --------------------

def _search_candidates(
    retr: Retriever,
    query: str,
    k: int,
    reranker: Optional[Reranker],
    rerank_topn: int,
    hybrid: bool = False,
    llm_reranker: Optional[GeminiReranker] = None,
) -> List[Dict[str, Any]]:
    need_overfetch = (reranker is not None) or (llm_reranker is not None)
    overfetch = max(5 * k, rerank_topn, k) if need_overfetch else k

    if hybrid:
        hits = retr.search_hybrid(query, k=overfetch, overfetch=overfetch)
    else:
        hits = retr.search(query, k=overfetch, overfetch=overfetch)

    if reranker:
        hits = reranker.rerank(query, hits, top_k=overfetch)

    if llm_reranker:
        llm_topn = min(len(hits), max(5 * k, rerank_topn))
        hits = llm_reranker.rerank(query, hits[:llm_topn], top_k=k)
    else:
        hits = hits[:k]

    return hits

# -------------------- main CLI --------------------

@app.command()
def main(
    index: str = typer.Option("data/index", "--index"),
    k: int = typer.Option(10, "--k"),
    device: Optional[str] = typer.Option(None, "--device"),
    hybrid: bool = typer.Option(False, "--hybrid"),
    rerank_model: Optional[str] = typer.Option(None, "--rerank-model"),
    rerank_topn: int = typer.Option(100, "--rerank-topn"),
    llm_rerank: bool = typer.Option(False, "--llm-rerank"),
    llm_model: str = typer.Option("gemini-2.5-flash", "--llm-model"),
    out_submission: str = typer.Option("reports/submission.json", "--out-submission"),
    out_report: str = typer.Option("reports/predictions_public.md", "--out-report"),
    ctx_hits: int = typer.Option(6, "--ctx-hits", help="сколько хитов давать в контекст LLM"),
    ctx_chars: int = typer.Option(800, "--ctx-chars", help="макс. символов на хит"),
    submit_chunks: int = typer.Option(2, "--submit-chunks", help="сколько ссылок класть в relevant_chunks"),
):
    """
    Делает сабмишн (с relevant_chunks) и отчёт сравнения с эталоном.
    """
    app_cfg = _load_app_cfg()
    dfq, dfa = _load_public(app_cfg)

    retr = Retriever.from_dir(Path(index), device=device)
    rr = Reranker(rerank_model, device=device) if rerank_model else None
    llm_rr = GeminiReranker(model=llm_model) if llm_rerank else None

    rows_md: List[str] = [
        "# Public QA — предсказания",
        "",
        "| qid | вопрос | предсказание | эталон | match | источник |",
        "|---:|---|---|---|:---:|---|",
    ]
    submission_items: List[Dict[str, Any]] = []

    # мапа эталона
    gold = {int(r["question_id"]): r["answer"] for r in dfa.to_dict(orient="records")}

    for row in dfq.to_dict(orient="records"):
        qid = int(row["question_id"])
        qtext = _light_norm(row["question"])
        atype = str(row.get("answer_type", "")).strip().lower()

        # 1) Ретрив кандидатов
        hits = _search_candidates(
            retr, qtext, k, rr, rerank_topn, hybrid=hybrid, llm_reranker=llm_rr
        )

        # 2) Контекст и извлечение ответа
        ctx = _build_context_snippets(hits, max_hits=ctx_hits, max_chars=ctx_chars)
        try:
            pred_text = _answer_from_context_gemini(qtext, ctx, atype, model_name=llm_model)
        except Exception as e:
            pred_text = f"N/A ({e})"

        pred_coerced = _coerce_answer(pred_text, atype)

        # 3) relevant_chunks из top-хитов
        rel_chunks = _choose_relevant_chunks(hits, max_chunks=submit_chunks)

        submission_items.append({
            "question_id": qid,
            "relevant_chunks": rel_chunks,
            "answer": pred_coerced,
        })

        # 4) сравнение с эталоном (для отчёта)
        gold_str = _light_norm(gold.get(qid, ""))
        pat_list = relaxed_answer_match.compile_patterns(gold_str) if gold_str else []
        ok = False
        if pat_list:
            ok, _ = relaxed_answer_match.any_match(str(pred_coerced), pat_list)

        src = f"{rel_chunks[0]['document_name']}:{rel_chunks[0]['page_number']}" if rel_chunks else ""
        rows_md.append(
            f"| {qid} | {qtext} | {pred_coerced} | {gold_str} | {'✅' if ok else '❌'} | {src} |"
        )

    # 5) Сабмишн
    out_submission_path = Path(out_submission)
    out_submission_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_submission_path, "w", encoding="utf-8") as f:
        json.dump(submission_items, f, ensure_ascii=False, indent=2)

    # 6) Markdown-отчёт
    out_report_path = Path(out_report)
    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    out_report_path.write_text("\n".join(rows_md), encoding="utf-8")

    typer.echo(f"✅ Saved submission to {out_submission_path.as_posix()}")
    typer.echo(f"📝 Saved report to {out_report_path.as_posix()}")

if __name__ == "__main__":  # pragma: no cover
    app()
