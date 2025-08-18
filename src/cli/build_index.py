import typer

from src.retrieval.embed import build_index

app = typer.Typer(add_completion=False)


@app.command()
def main(
    index: str = "data/index",
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
):
    build_index(index, model=model, device=device)
    typer.secho(f"index built at {index}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()

