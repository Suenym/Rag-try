import json, sys
from pathlib import Path
import pandas as pd
import typer
from src.validate.schema import load_schema, validate_submission


app = typer.Typer(add_completion=False)


@app.command()
def main(
    questions_path: str = "data/public/questions_public.xlsx",
    answers_path: str   = "data/public/answers_public.json",
    schema_path: str    = "configs/submission.schema.json",
):
    # 1) Проверяем, что вопросы читаются
    df_q = pd.read_excel(questions_path)
    assert "question_id" in df_q.columns, "questions_public.xlsx must contain 'question_id'"
    # 2) Проверяем, что публичные ответы соответствуют схеме
    answers = json.loads(Path(answers_path).read_text(encoding="utf-8"))
    schema = load_schema(schema_path)
    errors = validate_submission(answers, schema)
    if errors:
        typer.secho("Public answers do NOT match schema:", fg=typer.colors.RED)
        for e in errors[:50]:
            print(" -", e)
        sys.exit(1)
    typer.secho("OK: public answers match submission schema", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
