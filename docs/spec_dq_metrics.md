# Data Quality Metrics and Gates

This document lists metrics, thresholds and actions that protect data quality across the pipeline.

## OCR Metrics
- `cer_max`: **0.05**
- `suspect_char_ratio_max`: **0.03**
- `max_retry_profiles`: **2**

**Action**
- If `proxy_CER` ≤ 0.05 and `suspect_char_ratio` ≤ 0.03 → accept.
- Else try alternate profile until retries exhausted.
- After retries still failing → mark page for manual review.

## Table Metrics
- `struct_valid_min`: **0.7**
- `numeric_ratio_min`: **0.2**
- `header_required`: **true**

**Action**
- Scores above thresholds → accept.
- Otherwise attempt next extraction mode or flag `risky_table`.

## Normalization Metrics
- `require_sign_preserved`: **true**
- `forbid_unintended_rounding`: **true**

**Action**
- Violations lead to rejection and manual fix.

## Public Answer Findability
- Target `answer_findability_target`: **0.90** on public dataset.
- Achieved via manual checks; results recorded for next phase.

## Gate Matrix
| Metric | Threshold | Action |
|--------|-----------|--------|
| proxy-CER | ≤0.05 | accept page |
| proxy-CER | >0.05 | switch profile / manual review |
| suspect_char_ratio | ≤0.03 | accept |
| suspect_char_ratio | >0.03 | switch profile / manual review |
| table struct_valid | ≥0.7 | accept table |
| table struct_valid | <0.7 | next mode / risky_table |
| table numeric_ratio | ≥0.2 | accept |
| table numeric_ratio | <0.2 | risky_table |
| normalization sign | preserved | accept |
| normalization sign | lost | reject |
| normalization rounding | none | accept |
| normalization rounding | altered | reject |
| answer findability | ≥0.90 | goal met |
| answer findability | <0.90 | expand coverage |
