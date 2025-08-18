# Test Protocol on Public Dataset

## Inputs
- `data/public/questions_public.xlsx`
- `data/public/answers_public.json`
- `data/public/submission_format.json`
- Configs in `configs/`

## Procedure
1. **Layout Check**
   - Run `python -m src.cli.check_public_layout`.
   - Confirm files exist, `question_id` column present, answers JSON is an array.
2. **Schema Validation**
   - Run `python -m src.cli.validate_public` to ensure answers follow `submission.schema.json`.
   - Check that numeric answers use correct decimal separators and units.
3. **Typing of Answers**
   - For each sample question verify that the answer type (number or string) matches schema expectations.
4. **Answer Findability Prep**
   - For every question, note keywords that should appear in source documents.
   - In future stages manually verify that processed documents contain these answers to reach the target `answer_findability_target`.
5. **Report Generation**
   - Expect the following reports: `dq_ocr_report.md`, `dq_tables_report.md`, `dq_norm_report.md`, `dq_summary.md` in the `reports/` folder.
6. **Final Checklist**
   - Fill out `reports/dq_summary.md` and mark each item as accepted or not.
   - Store any logs in `logs/`.

## Outputs
- Validated public dataset.
- Filled `dq_summary.md` with final decision `Принято/Не принято`.
