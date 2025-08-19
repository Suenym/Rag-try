# src/cli/search.py
from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer

from src.retrieval.search import Retriever
from src.retrieval.rerank import Reranker

app = typer.Typer(add_completion=False)

@app.command()
def main(
    index: str = typer.Option("data/index", "--index"),
    q: str = typer.Option(..., "--q"),
    k: int = typer.Option(5, "--k"),
    device: Optional[str] = typer.Option(None, "--device", help="cpu | cuda"),
    rerank_model: Optional[str] = typer.Option(None, "--rerank-model", help="e.g. cross-encoder/ms-marco-MiniLM-L-6-v2"),
    rerank_topn: int = typer.Option(50, "--rerank-topn", help="сколько кандидатов переранжировать (до k*10 имеет смысл)"),
):
    retr = Retriever.from_dir(Path(index), device=device)

    # если есть переранжировка — вытаскиваем больше кандидатов
    fetch_k = max(k, k * 5, rerank_topn if rerank_model else 0)
    hits = retr.search(q, k=fetch_k)

    if rerank_model:
        rr = Reranker(rerank_model, device=device)
        hits = rr.rerank(q, hits, top_k=k)
    else:
        hits = hits[:k]

    for h in hits:
        print(f"score={h.get('rerank_score', h['score']):.3f} | {h['doc_name']}:{h['page_number']} | {h['kind']} | {h['preview']}")

if __name__ == "__main__":
    app()
