from __future__ import annotations

import re
from typing import Any, List, Tuple, Optional

# Relaxed matching helpers copied from answer_findability_public

QUOTES_CLASS = r'["\'«»„“”‟`ʼ’]'
_WS = r'(?:[ \t\u00A0\u202F]|\n|\r)+'

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
    s0 = _norm_simple(s)
    s0 = (
        s0.replace("«", '")').replace("»", '")')
           .replace("“", '")').replace("”", '")')
           .replace("„", '")').replace("‟", '")')
           .replace("’", "'")
    )
    esc = re.escape(s0)
    esc = esc.replace(r"\"", QUOTES_CLASS).replace(r"\'", QUOTES_CLASS)
    esc = re.sub(r"\\\s\+", lambda _m: _WS, esc)
    esc = esc.replace(r"\ ", _WS)
    esc = esc.replace(r"\-", r"[\-–—]")
    esc = re.sub(
        r"(?<=\\w)\\[.,](?=\\w)",
        lambda _m: r"(?:[.,]?\s*)",
        esc,
    )
    esc = esc.replace(_WS + _WS, _WS)
    return esc


def _int_to_regex(n: int) -> str:
    s = str(abs(int(n)))
    sep = r'(?:[ \u00A0\u202F]|[,\.])'
    if len(s) <= 3:
        body = _digits_ocr_friendly(s)
    else:
        rem = len(s) % 3 or 3
        groups = [s[:rem]] + [s[i:i+3] for i in range(rem, len(s), 3)]
        body = _digits_ocr_friendly(groups[0]) + "".join(
            f'(?:{sep}?{_digits_ocr_friendly(g)})' for g in groups[1:]
        )
    prefix = r'(?:\bБИН\s+|[₸]?\s*)?'
    return rf'(?<!\d){prefix}{body}(?!\d)'


def _float_to_regex(x: float) -> str:
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


def build_answer_patterns(answer: Any) -> List[Tuple[str, bool]]:
    pats: List[Tuple[str, bool]] = []
    if isinstance(answer, bool):
        pats.append((rf"(?<!\w){str(answer).lower()}(?!\w)", True))
    elif isinstance(answer, (int,)) and not isinstance(answer, bool):
        pats.append((_int_to_regex(int(answer)), False))
    elif isinstance(answer, float):
        pats.append((_float_to_regex(float(answer)), False))
        pats.append((_int_to_regex(int(round(float(answer)))), False))
    else:
        s = str(answer)
        pats.append((_string_to_relaxed_regex(s), True))
        s2 = re.sub(QUOTES_CLASS, "", s)
        if s2 != s:
            pats.append((_string_to_relaxed_regex(s2), True))
    return pats


def match_answer(text: str, answer: Any) -> Optional[str]:
    patterns = build_answer_patterns(answer)
    for rx, ci in patterns:
        flags = re.IGNORECASE if ci else 0
        try:
            if re.search(rx, text, flags=flags):
                return rx
        except re.error:
            if re.search(re.escape(rx), text, flags=flags):
                return rx
    return None

