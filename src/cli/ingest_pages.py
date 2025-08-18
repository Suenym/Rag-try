import typer

from src.ingest.pdf_pages import ingest_pdfs

app = typer.Typer()


@app.command()
def main(in_: str = typer.Option(..., "--in"), out: str = typer.Option(..., "--out")) -> None:
    ingest_pdfs(in_, out)


if __name__ == "__main__":
    app()
