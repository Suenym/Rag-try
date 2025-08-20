from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Tuple
import re


@dataclass
class Answer:
    value: Any
    atype: str
    matched_hit_idx: Optional[int]  # индекс в списке hits, где найдено
    matched_span: Optional[Tuple[int, int]]


MONTHS_RU = r"(январ[ья]|феврал[ья]|март[ае]?|апрел[ья]|ма[йя]|июн[ья]|июл[ья]|август[ае]?|сентябр[ья]|октябр[ья]|ноябр[ья]|декабр[ья])"


def detect_answer_type(q: str) -> str:
    ql = q.lower()
    if "кто" in ql or "фамили" in ql or "бухгалтер" in ql or "директор" in ql or "должност" in ql or "как называется" in ql or "какой организацией" in ql:
        return "string"
    if "%" in ql or "процент" in ql or "в %" in ql:
        return "percent"  # будет float
    if "дата" in ql or re.search(r"\b(год|году|месяц)\b", ql):
        return "int"  # чаще целое (годы/месяцы)
    # по умолчанию число
    return "float" if re.search(r"(значен|показател|тонн|частот|объём|объем)", ql) else "int"


# Регэкспы
RE_INT = re.compile(r"(?<![\d.,])\d{1,3}(?:[ \u00A0]\d{3})+|\d+")
RE_FLOAT = re.compile(r"(?<![\d.,])\d{1,3}(?:[ \u00A0]\d{3})+[.,]\d+|\d+[.,]\d+")
RE_PERCENT = re.compile(r"(?<![\d.,])\d{1,3}(?:[ \u00A0]\d{3})*[.,]?\d*\s?%|\d+[.,]?\d*\s?%")
RE_DATE = re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{4}\b|\b\d{1,2}\s+" + MONTHS_RU + r"\s+\d{4}\b", re.IGNORECASE)
RE_NAME = re.compile(r"\b[А-ЯЁ][а-яё\-]+(?:\s+[А-ЯЁ][а-яё\-]+){1,2}\b")


def _first_match(pat: re.Pattern, text: str):
    m = pat.search(text)
    if m:
        return m.group(0), (m.start(), m.end())
    return None, None


def extract_answer_from_hits(question: str, hits: List[Dict], topk: int = 3) -> Answer:
    atype = detect_answer_type(question)
    text_pool = []
    idx_map = []
    for i, h in enumerate(hits[:topk]):
        t = str(h.get("preview", "") or "")
        if t:
            text_pool.append(t)
            idx_map.append(i)
    big = "\n\n".join(text_pool)

    # порядок поиска
    if atype == "percent":
        val, span = _first_match(RE_PERCENT, big)
        if val:
            return Answer(value=_fix_percent(val), atype="float", matched_hit_idx=_span2hit(span, text_pool, idx_map), matched_span=span)
    if atype in ("float",):
        val, span = _first_match(RE_FLOAT, big)
        if val:
            return Answer(value=_fix_number(val), atype="float", matched_hit_idx=_span2hit(span, text_pool, idx_map), matched_span=span)
    if atype in ("int",):
        val, span = _first_match(RE_INT, big)
        if val:
            return Answer(value=_fix_int(val), atype="int", matched_hit_idx=_span2hit(span, text_pool, idx_map), matched_span=span)
    if atype == "string":
        # простая эвристика: сначала имя/ФИО, иначе кавычки
        val, span = _first_match(RE_NAME, big)
        if not val:
            m = re.search(r"«([^»]{2,80})»|\"([^\"\n]{2,80})\"", big)
            if m:
                val = m.group(1) or m.group(2)
                span = (m.start(), m.end())
        if val:
            return Answer(value=val.strip(), atype="string", matched_hit_idx=_span2hit(span, text_pool, idx_map), matched_span=span)

    return Answer(value="N/A", atype=atype, matched_hit_idx=None, matched_span=None)


def _fix_number(s: str) -> float:
    s = s.replace("\u00A0", " ").replace(" ", "").replace(",", ".")
    return float(s)


def _fix_int(s: str) -> int:
    s = s.replace("\u00A0", " ").replace(" ", "")
    s = s.split(",")[0].split(".")[0]
    return int(s)


def _fix_percent(s: str) -> float:
    s = s.replace("%", "").strip()
    s = s.replace("\u00A0", " ").replace(" ", "").replace(",", ".")
    return float(s)


def _span2hit(span, texts: List[str], idx_map: List[int]) -> Optional[int]:
    if not span:
        return None
    start, _ = span
    pos = 0
    for t, idx in zip(texts, idx_map):
        end = pos + len(t)
        if pos <= start < end:
            return idx
        pos = end + 2  # "\n\n"
    return None
