import typer

from src.retrieval.search import SearchIndex

app = typer.Typer(add_completion=False)


@app.command()
def main(
    index: str = "data/index",
    q: str = typer.Option(..., help="query string"),
    k: int = 5,
):
    searcher = SearchIndex(index)
    res = searcher.search(q, k=k)
    if res.empty:
        typer.echo("No results")
    else:
        for r in res.itertuples():
            typer.echo(
                f"score={r.score:.3f} | {r.doc_name}:{r.page_number} | {r.source_type} | {r.text[:80]}"
            )


if __name__ == "__main__":
    app()

