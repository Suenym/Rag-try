import typer

from src.ocr.run_ocr import run_ocr
from src.tables.extract_tables import extract_tables
from src.normalize.apply_normalization import apply_normalization
from src.validate.run_dq_gates import run_dq_gates

app = typer.Typer()


@app.command()
def main(cache: str = typer.Option("data/cache", "--cache")) -> None:
    run_ocr(cache)
    extract_tables(cache)
    apply_normalization(cache)
    run_dq_gates(cache)


if __name__ == "__main__":
    app()
