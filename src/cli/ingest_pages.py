from __future__ import annotations

import typer
from src.ingest.pdf_pages import ingest_pdfs

app = typer.Typer(add_completion=False)

@app.command()
def main(in_: str = typer.Option(..., "--in", help="Input PDF folder"),
         out: str = typer.Option("data/cache", "--out", help="Cache/output folder")):
    ingest_pdfs(in_, out)

if __name__ == "__main__":  # pragma: no cover
    app()
