# Normalization Specification

## Text Rules
- **Hyphenated line breaks** are joined when the next line continues with a lowercase letter (`hyphen_fix_enabled: true`).
- Collapse multiple spaces and trim leading/trailing whitespace.
- Preserve sign characters; dropping `+` or `-` is forbidden.

## Numbers
- Thousand separators allowed: space (`" "`) and non‑breaking space (`\u00A0`).
- Decimal separators: comma and dot.
- Scientific notation is disallowed (`allow_scientific_notation: false`).
- Percent values must contain `%` (`accept_percent_without_symbol: false`).
- Currency symbols (`₽`, `₸`, `$`, `€`) are kept next to the number.
- Conversion may not cause unintended rounding (`forbid_unintended_rounding: true`).

## Dates
- Accept formats: `dd.mm.yyyy`, `dd-mm-yyyy`, `yyyy-mm-dd` (RU/KZ styles).
- Normalize to ISO `yyyy-mm-dd`.
- Validate year range 1900–2100.

## Logging
For every transformation record in log:
- field name
- original value
- normalized value
- rule applied
- any ambiguity notes

## Ambiguities
- Flag numbers where sign cannot be determined (`require_sign_preserved: true`).
- Flag dates with ambiguous ordering or missing year.
