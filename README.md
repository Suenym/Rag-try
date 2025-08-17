# Offline RAG OCR — Baseline (Bootstrap)

Цель: жёсткая фиксация формата сабмита, валидатор JSON, первичная интеграция публичных данных, DQ-отчёт.

## Быстрый старт
```bash
# 1) Установка
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Положите файлы:
#   data/public/questions_public.xlsx
#   data/public/answers_public.json
#   data/public/submission_format.json

# 3) Валидация и генерация черновика сабмита
python -m src.cli.validate_public
python -m src.cli.make_draft_submission --out submission/submission.json
```

Стандарты

Сабмит строго следует JSON-схеме (см. configs/submission.schema.json).

question_id: int; relevant_chunks: non-empty; answer: (number|string).

Ссылки: {document_name: str, page_number: int (1-based)}.
