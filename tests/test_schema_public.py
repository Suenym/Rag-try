import json
from pathlib import Path
from src.validate.schema import load_schema, validate_submission


def test_public_answers_match_schema():
    answers = json.loads(Path("data/public/answers_public.json").read_text(encoding="utf-8"))
    schema = load_schema("configs/submission.schema.json")
    errors = validate_submission(answers, schema)
    assert not errors, f"Schema errors: {errors[:5]}"
