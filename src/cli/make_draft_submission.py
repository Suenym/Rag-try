import json
from pathlib import Path
import pandas as pd
import typer

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
    out_path: str = "submission/submission.json"
):
    df = pd.read_excel(questions_path)
    id_col = pick_id_col(df.columns)
    if not id_col:
        raise SystemExit(
            f"questions_public.xlsx must contain one of {CANDIDATE_ID_COLS}; "
            f"found: {list(df.columns)}"
        )
    if id_col != "question_id":
        df = df.rename(columns={id_col: "question_id"})

    draft = []
    for _, row in df.iterrows():
        try:
            qid = int(row["question_id"])
        except Exception:
            # если встречается пустое/нецелочисленное — пропустим такую строку
            continue
        draft.append({
            "question_id": qid,
            "relevant_chunks": [
                {"document_name": "TBD.pdf", "page_number": 1}
            ],
            "answer": "TBD"
        })
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[make_draft_submission] Used id column: {id_col}")
    print(f"Draft submission saved to {out_path}")

if __name__ == "__main__":
    app()
