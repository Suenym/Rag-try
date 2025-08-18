import typer

from src.retrieval.chunk import build_chunks

app = typer.Typer(add_completion=False)


@app.command()
def main(
    cache: str = "data/cache",
    out: str = "data/index",
    page_chunk_chars: int = 1200,
    overlap_chars: int = 200,
):
    build_chunks(cache, out, page_chunk_chars, overlap_chars)
    typer.secho(f"chunks saved to {out}/chunks.parquet", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()

