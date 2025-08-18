from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import re

import pandas as pd
import yaml

# Try optional fuzzy lib (faster & better). Falls back to difflib.
try:
    from rapidfuzz import fuzz  # type: ignore
    _HAVE_RAPIDFUZZ = True
except Exception:
    import difflib
    _HAVE_RAPIDFUZZ = False


def _load_app_cfg() -> dict:
    with open("configs/app.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# -----------------------------
# Normalizers & pattern builders
# -----------------------------
QUOTES_CLASS = r'["\'«»„“”‟`ʼ’]'
_WS = r'(?:[ \t\u00A0\u202F]|\n|\r)+'  # space, tab, NBSP, narrow NBSP, or line breaks

# OCR-friendly digit class mapping (handles common confusions)
_DIGIT_MAP = {
    "0": "[0Oo]",
    "1": "[1IlІi|]",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "[5Ss]",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
}

def _digits_ocr_friendly(s: str) -> str:
    return "".join(_DIGIT_MAP.get(ch, re.escape(ch)) for ch in s)

def _norm_simple(s: str) -> str:
    s = (s or "").replace("\u00A0", " ").replace("\u202F", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _string_to_relaxed_regex(s: str) -> str:
    """
    Tolerant to:
      - quote glyph variants («» “” ‘’ ' ")
      - variable whitespace (space / NBSP / narrow NBSP / newlines)
      - dash variants (- – —)
      - light punctuation between word chars (e.g., 'Co.LTD' vs 'Co LTD')
    """
    s0 = _norm_simple(s)
    s0 = (
        s0.replace("«", '"').replace("»", '"')
           .replace("“", '"').replace("”", '"')
           .replace("„", '"').replace("‟", '"')
           .replace("’", "'")
    )
    esc = re.escape(s0)

    # Relax quotes
    esc = esc.replace(r"\"", QUOTES_CLASS).replace(r"\'", QUOTES_CLASS)

    # Relax whitespace (use callable to avoid \u / \s escaping issues)
    esc = re.sub(r"\\\s\+", lambda _m: _WS, esc)
    esc = esc.replace(r"\ ", _WS)

    # Relax hyphens/dashes
    esc = esc.replace(r"\-", r"[\-–—]")

    # Relax light punctuation between word chars:
    # Replace literal '\.' or '\,' that are between word chars with a pattern
    # that allows optional dot/comma and optional space.
    esc = re.sub(
        r"(?<=\\w)\\[.,](?=\\w)",
        lambda _m: r"(?:[.,]?\s*)",
        esc,
    )

    # De-duplicate accidental doubled WS tokens
    esc = esc.replace(_WS + _WS, _WS)
    return esc


def _int_to_regex(n: int) -> str:
    """
    Integer tolerant to optional thousands separators (space, NBSP, narrow NBSP, comma, dot),
    OCR digit confusions, and optional currency/label prefix.
    """
    s = str(abs(int(n)))
    sep = r'(?:[ \u00A0\u202F]|[,\.])'  # space, NBSP, thin NBSP, comma, dot

    if len(s) <= 3:
        body = _digits_ocr_friendly(s)
    else:
        rem = len(s) % 3 or 3
        groups = [s[:rem]] + [s[i:i+3] for i in range(rem, len(s), 3)]
        body = _digits_ocr_friendly(groups[0]) + "".join(
            f'(?:{sep}?{_digits_ocr_friendly(g)})' for g in groups[1:]
        )

    prefix = r'(?:\bБИН\s+|[₸]?\s*)?'  # tolerate labels like БИН / currency mark
    return rf'(?<!\d){prefix}{body}(?!\d)'

def _float_to_regex(x: float) -> str:
    """
    Decimal tolerant to:
      - , or . as decimal separator
      - optional thousands separators in the integer part
      - trailing zeros in the fractional part
      - OCR digit confusions
    """
    x = float(x)
    if x.is_integer():
        return _int_to_regex(int(x))

    neg = x < 0
    s = f"{abs(x):f}".rstrip("0").rstrip(".")
    whole, frac = s.split(".", 1)

    sep = r'(?:[ \u00A0\u202F]|[,\.])'
    if len(whole) <= 3:
        whole_body = _digits_ocr_friendly(whole)
    else:
        rem = len(whole) % 3 or 3
        groups = [whole[:rem]] + [whole[i:i+3] for i in range(rem, len(whole), 3)]
        whole_body = _digits_ocr_friendly(groups[0]) + "".join(
            f'(?:{sep}?{_digits_ocr_friendly(g)})' for g in groups[1:]
        )

    frac_body = re.escape(frac) + r'(?:0+)?'
    sign = r'\-' if neg else ''
    return rf'(?<!\d){sign}{whole_body}[,.]{frac_body}(?!\d)'


# -----------------------------
# Candidate selection & search
# -----------------------------
def _select_candidates(page_df: pd.DataFrame, rc: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rc:
        return page_df
    key = pd.DataFrame([{
        "doc_name": r["document_name"],
        "page_number": int(r["page_number"]),
    } for r in rc])

    # Include ±1 pages to absorb small indexing shifts
    plusminus = pd.concat([
        key.assign(page_number=key.page_number - 1),
        key,
        key.assign(page_number=key.page_number + 1),
    ], ignore_index=True).drop_duplicates()

    return page_df.merge(plusminus, on=["doc_name", "page_number"], how="inner")


def _build_patterns(answer: Any) -> List[Tuple[str, bool]]:
    """
    Return list of (regex, case_insensitive) patterns to test in order.
    """
    pats: List[Tuple[str, bool]] = []
    if isinstance(answer, bool):
        pats.append((rf"(?<!\w){str(answer).lower()}(?!\w)", True))
    elif isinstance(answer, (int,)) and not isinstance(answer, bool):
        pats.append((_int_to_regex(int(answer)), False))
    elif isinstance(answer, float):
        pats.append((_float_to_regex(float(answer)), False))
        # also consider when someone wrote integer-rounded
        pats.append((_int_to_regex(int(round(float(answer)))), False))
    else:
        s = str(answer)
        # exact relaxed string
        pats.append((_string_to_relaxed_regex(s), True))
        # without quotes fully
        s2 = re.sub(QUOTES_CLASS, "", s)
        if s2 != s:
            pats.append((_string_to_relaxed_regex(s2), True))
    return pats


def _try_regex_hit(df: pd.DataFrame, text_col: str, patterns: List[Tuple[str, bool]]) -> Optional[Tuple[str, int]]:
    if df.empty or text_col not in df:
        return None
    text = df[text_col].fillna("").astype(str)
    for rx, ci in patterns:
        try:
            mask = text.str.contains(rx, case=ci, regex=True)
        except re.error:
            mask = text.str.contains(re.escape(rx), case=ci, regex=True)
        if mask.any():
            row = df.loc[mask].iloc[0]
            return (row["doc_name"], int(row["page_number"]))
    return None


def _prep_for_fuzzy(s: str) -> str:
    s = (s or "")
    s = s.replace("\u00A0", " ").replace("\u202F", " ")
    s = re.sub(QUOTES_CLASS, "", s)
    s = re.sub(r"[\-–—]", "-", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def _try_fuzzy_hit(df: pd.DataFrame, text_col: str, answer: Any, threshold: int = 85) -> Optional[Tuple[str, int]]:
    if df.empty or text_col not in df:
        return None
    # Only apply fuzzy to string-like answers of reasonable length
    s = str(answer)
    s_clean = _prep_for_fuzzy(s)
    if len(s_clean) < 6:
        return None

    series = df[text_col].fillna("").astype(str)
    for _, row in df.iterrows():
        t = _prep_for_fuzzy(str(row[text_col]))
        if not t:
            continue
        if _HAVE_RAPIDFUZZ:
            score = fuzz.partial_ratio(s_clean, t)
            if score >= threshold:
                return (row["doc_name"], int(row["page_number"]))
        else:
            # cheap difflib fallback: check a few slices
            if s_clean in t:
                return (row["doc_name"], int(row["page_number"]))
            # Very light sliding check around occurrences of the first 6 chars
            key = s_clean[:6]
            for m in re.finditer(re.escape(key), t):
                start = max(0, m.start() - 20)
                end = min(len(t), m.start() + len(s_clean) + 20)
                window = t[start:end]
                ratio = (2.0 * len(set(window) & set(s_clean))) / (len(window) + len(s_clean))
                if ratio >= 0.6:
                    return (row["doc_name"], int(row["page_number"]))
    return None


def _search_everywhere(
    page_df: pd.DataFrame,
    table_df: Optional[pd.DataFrame],
    rc: List[Dict[str, Any]],
    patterns: List[Tuple[str, bool]],
    answer: Any,
) -> Optional[Tuple[str, int, str]]:
    """
    Try candidates first (pages, then tables), then whole corpus.
    Returns (doc_name, page_number, source) or None.
    """
    cand_pages = _select_candidates(page_df, rc)
    hit = _try_regex_hit(cand_pages, "text_norm", patterns)
    if hit:
        return (*hit, "page")

    if table_df is not None and not table_df.empty:
        if rc:
            key = pd.DataFrame([{"doc_name": r["document_name"], "page_number": int(r["page_number"])} for r in rc])
            cand_tables = table_df.merge(key, on=["doc_name", "page_number"], how="inner")
        else:
            cand_tables = table_df

        for col in [c for c in ["row_text_norm", "header_text_norm"] if c in cand_tables.columns]:
            hit = _try_regex_hit(cand_tables, col, patterns)
            if hit:
                return (*hit, "table:"+col)

    # Fuzzy fallback on candidates (limits cost, fixes OCR typos like Logisnics/Logistics)
    fuzzy = _try_fuzzy_hit(cand_pages, "text_norm", answer)
    if fuzzy:
        return (*fuzzy, "page_fuzzy")

    if table_df is not None and not table_df.empty:
        if rc:
            key = pd.DataFrame([{"doc_name": r["document_name"], "page_number": int(r["page_number"])} for r in rc])
            cand_tables = table_df.merge(key, on=["doc_name", "page_number"], how="inner")
        else:
            cand_tables = table_df
        for col in [c for c in ["row_text_norm", "header_text_norm"] if c in cand_tables.columns]:
            fuzzy = _try_fuzzy_hit(cand_tables, col, answer)
            if fuzzy:
                return (*fuzzy, "table_fuzzy:"+col)

    # Last resort: global regex
    hit = _try_regex_hit(page_df, "text_norm", patterns)
    if hit:
        return (*hit, "page_global")

    if table_df is not None and not table_df.empty:
        for col in [c for c in ["row_text_norm", "header_text_norm"] if c in table_df.columns]:
            hit = _try_regex_hit(table_df, col, patterns)
            if hit:
                return (*hit, "table_global:"+col)

    # Last resort fuzzy on whole corpus pages (can be slow; keep as final safety)
    fuzzy = _try_fuzzy_hit(page_df, "text_norm", answer)
    if fuzzy:
        return (*fuzzy, "page_global_fuzzy")

    if table_df is not None and not table_df.empty:
        for col in [c for c in ["row_text_norm", "header_text_norm"] if c in table_df.columns]:
            fuzzy = _try_fuzzy_hit(table_df, col, answer)
            if fuzzy:
                return (*fuzzy, "table_global_fuzzy:"+col)

    return None


# -----------------------------
# Public API
# -----------------------------
def answer_findability(cache_dir: str) -> Dict[str, Any]:
    app_cfg = _load_app_cfg()
    answers_path = Path(app_cfg["paths"]["public_answers"])
    with open(answers_path, "r", encoding="utf-8") as f:
        answers = json.load(f)

    page_df = pd.read_parquet(Path(cache_dir) / "page_text_norm.parquet")  # doc_name,page_number,text_norm
    tpath = Path(cache_dir) / "tables_norm.parquet"
    table_df = pd.read_parquet(tpath) if tpath.exists() else None

    found, misses = [], []
    details: List[str] = []

    for item in answers:
        qid = int(item["question_id"])
        ans = item["answer"]
        rc = item.get("relevant_chunks", [])

        patterns = _build_patterns(ans)
        hit = _search_everywhere(page_df, table_df, rc, patterns, ans)

        if hit:
            found.append(qid)
            dname, pnum, src = hit
            details.append(f"- q{qid} {dname} p{pnum} ({src})")
        else:
            misses.append(qid)

    coverage = (len(found) / len(answers)) if answers else 0.0

    # Write a small report
    report = [
        f"# Public Findability Report",
        f"- Total Q: {len(answers)}",
        f"- Found  : {len(found)} ({coverage:.2%})",
        f"- Missed : {len(misses)}",
        "",
        f"**Found IDs**: {sorted(found)}",
        f"**Missed IDs**: {sorted(misses)}",
    ]
    if details:
        report += ["", *details]

    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/public_findability.md").write_text("\n".join(report), encoding="utf-8")

    target = float(app_cfg.get("dq", {}).get("public", {}).get("answer_findability_target", 0.9))
    return {"coverage": coverage, "found": found, "missed": misses, "target": target}
