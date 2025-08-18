import typer

from src.validate.answer_findability_public import answer_findability

app = typer.Typer()


@app.command()
def main(cache: str = typer.Option("data/cache", "--cache")) -> None:
    metrics = answer_findability(cache)
    typer.echo(f"Coverage: {metrics['coverage']:.2%} (target {metrics['target']:.0%})")


if __name__ == "__main__":
    app()
