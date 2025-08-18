# Table Extraction Policy

## Extraction Order
Preferred modes in decreasing priority:
1. **Lattice** – uses ruling lines.
2. **Stream** – text stream heuristics.
3. **Scan** – vision-based detection.

Always attempt modes sequentially: **Lattice → Stream → Scan**.

## Validity Criteria
- `struct_valid_min`: **0.7** minimal structure confidence.
- `numeric_ratio_min`: **0.2** minimal share of numeric cells.
- `header_required`: **true**.

If a mode yields metrics below thresholds or header is missing:
- Attempt the next mode.
- After all modes, mark table as `risky_table` and log.

## Risky Table Flag
`risky_table = true` when any of the following is observed:
- structural validity within 10% of threshold.
- numeric ratio < `numeric_ratio_min`.
- header missing after all modes.
- non-rectangular column alignment or merged cells.

## Metadata
For each detected table store:
- `page_id`, `mode_used`.
- `struct_valid_score`, `numeric_ratio`, `header_present`.
- bounding boxes, row/column counts.
- `risky_table` flag.
