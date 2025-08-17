import json
from jsonschema import Draft202012Validator
from pathlib import Path


def load_schema(path: str) -> Draft202012Validator:
    schema = json.loads(Path(path).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def validate_submission(submission_obj, schema_validator: Draft202012Validator) -> list[str]:
    errors = []
    for err in schema_validator.iter_errors(submission_obj):
        errors.append(err.message)
    return errors
