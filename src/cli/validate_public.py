import json, sys
from pathlib import Path
import pandas as pd
import typer
from src.validate.schema import load_schema, validate_submission

app = typer.Typer(add_completion=False)

CANDIDATE_ID_COLS = ["question_id", "id", "qid", "QuestionID", "questionId"]

def pick_id_col(cols):
    colmap = {c.lower(): c for c in cols}
    for cand in CANDIDATE_ID_COLS:
        if cand.lower() in colmap:
            return colmap[cand.lower()]
    return None

@app.command()
def main(
    questions_path: str = "data/public/questions_public.xlsx",
    answers_path: str   = "data/public/answers_public.json",
    schema_path: str    = "configs/submission.schema.json",
):
    # 1) Проверяем, что вопросы читаются
    df_q = pd.read_excel(questions_path)
    id_col = pick_id_col(df_q.columns)
    if not id_col:
        print("ERROR: questions_public.xlsx must contain one of columns:",
              CANDIDATE_ID_COLS, "\nFound columns:", list(df_q.columns), file=sys.stderr)
        sys.exit(1)
    # Приводим к единому имени для дальнейших шагов
    if id_col != "question_id":
        df_q = df_q.rename(columns={id_col: "question_id"})
    # 2) Проверяем, что публичные ответы соответствуют схеме
    answers = json.loads(Path(answers_path).read_text(encoding="utf-8"))
    schema = load_schema(schema_path)
    errors = validate_submission(answers, schema)
    if errors:
        typer.secho("Public answers do NOT match schema:", fg=typer.colors.RED)
        for e in errors[:50]:
            print(" -", e)
        sys.exit(1)
    typer.secho(
        f"OK: public answers match submission schema | questions id column = '{id_col}'",
        fg=typer.colors.GREEN
    )

if __name__ == "__main__":
    app()
