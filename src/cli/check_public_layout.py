import json
from pathlib import Path

import pandas as pd
import typer
import yaml

app = typer.Typer(add_completion=False)


def load_config(path: str):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


@app.command()
def main(config_path: str = "configs/app.yaml"):
    cfg = load_config(config_path)
    problems = []

    paths = cfg.get("paths", {})
    q_path = Path(paths.get("public_questions", ""))
    a_path = Path(paths.get("public_answers", ""))

    if not q_path.exists():
        problems.append(f"Missing questions file: {q_path}")
    else:
        try:
            df = pd.read_excel(q_path, nrows=0)
            if "question_id" not in df.columns:
                problems.append("questions_public.xlsx must contain column 'question_id'")
        except Exception as e:
            problems.append(f"Failed to read questions file: {e}")

    if not a_path.exists():
        problems.append(f"Missing answers file: {a_path}")
    else:
        try:
            data = json.loads(a_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                problems.append("answers_public.json must be a JSON array")
        except Exception as e:
            problems.append(f"Invalid answers file: {e}")

    if problems:
        typer.secho("CHECKS: FAIL", fg=typer.colors.RED)
        for p in problems:
            print("-", p)
        raise typer.Exit(code=1)
    else:
        typer.secho("CHECKS: OK", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
