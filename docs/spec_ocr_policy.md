# OCR Policy Specification

## Supported Languages
- OCR operates with joint language model: **rus + kaz**.

## Profiles
| Profile | psm | oem | Use case |
|--------|-----|-----|----------|
| A | 6 | LSTM | Default profile, balanced layout |
| B | 4 | LSTM | Fallback when A produces poor quality |

## Workflow
1. Run **profile A** on every page flagged for OCR.
2. Compute proxy metrics: `proxy_CER` and `suspect_char_ratio`.
3. If `proxy_CER ≤ 0.05` **and** `suspect_char_ratio ≤ 0.03` → accept result.
4. Otherwise retry with **profile B**.
5. After trying both profiles (`max_retry_profiles = 2`) and still failing thresholds → mark page as `ocr_fail` and route to manual review.

## Fallback Rules
- Only one retry with profile B per page.
- Pages exceeding thresholds after both profiles are logged for escalation.

## Proxy Metrics
- **proxy_CER**: heuristic character error rate derived from language statistics.
- **suspect_char_ratio**: share of characters from a predefined suspect set (`�`, unprintables, etc.).

## Parallelism
- OCR runs with **parallelism = 2** workers to balance speed and determinism.

## Logging Fields
For each page the following fields are logged:
- `page_id`
- `profile_used`
- `languages`
- `proxy_CER`
- `suspect_char_ratio`
- `retry_count`
- `status` (ok/fail)
- timestamp
