import json
from pathlib import Path
import pandas as pd
import typer


app = typer.Typer(add_completion=False)


@app.command()
def main(
    questions_path: str = "data/public/questions_public.xlsx",
    out_path: str = "submission/submission.json"
):
    df = pd.read_excel(questions_path)
    required_cols = {"question_id"}
    missing = required_cols - set(df.columns)
    assert not missing, f"missing columns in questions_public.xlsx: {missing}"

    draft = []
    for _, row in df.iterrows():
        qid = int(row["question_id"])
        draft.append({
            "question_id": qid,
            "relevant_chunks": [
                {"document_name": "TBD.pdf", "page_number": 1}
            ],
            "answer": "TBD"
        })
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Draft submission saved to {out_path}")


if __name__ == "__main__":
    app()
