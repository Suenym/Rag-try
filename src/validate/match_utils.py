# src/validate/match_utils.py
from __future__ import annotations
import re
from typing import List, Tuple, Optional, Union, Iterable

_SEP_THIN = "\u202F"   # узкий неразрывный пробел
_SEP_NBSP = "\u00A0"   # nbsp
_SEP_GROUP = r"[ \u00A0\u202F,]"  # пробел | nbsp | узкий пробел | запятая

def _light_norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = (s
         .replace("«", '"').replace("»", '"')
         .replace("“", '"').replace("”", '"')
         .replace("„", '"')
         .replace(_SEP_NBSP, " ").replace(_SEP_THIN, " "))
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _is_numeric_string(s: str) -> bool:
    s = s.strip()
    # допускаем группы тысяч и десятичную часть
    return bool(re.fullmatch(r"\d{1,3}(?:[ \u00A0\u202F]\d{3})*(?:[.,]\d+)?|\d+(?:[.,]\d+)?", s))

def _split_int_triads(digits: str) -> Tuple[str, int]:
    """возвращает (первая_группа, кол-во триад) для 120309860000 -> ('120', 3)"""
    n = len(digits)
    k = (n - 1) // 3  # сколько полномерных триад после первой группы
    first_len = n - k * 3
    return digits[:first_len], k

def _regex_for_int(digits: str) -> str:
    # без разделителей
    plain = re.escape(digits)
    # с разделителями тысяч
    first, k = _split_int_triads(digits)
    with_grouping = rf"{re.escape(first)}(?:{_SEP_GROUP}\d{{3}}){{{k}}}"
    return rf"(?<!\d)(?:{with_grouping}|{plain})(?!\d)"

def _regex_for_float(int_part: str, frac_part: str) -> str:
    # integer часть может быть с группировкой, десятичная - точка или запятая
    first, k = _split_int_triads(int_part)
    int_grouped = rf"{re.escape(first)}(?:{_SEP_GROUP}\d{{3}}){{{k}}}"
    int_plain = re.escape(int_part)
    int_regex = rf"(?:{int_grouped}|{int_plain})"
    # допускаем '.' или ',' как разделитель
    return rf"(?<!\d){int_regex}[.,]{re.escape(frac_part)}(?!\d)"

def _numeric_patterns_from_str(s: str) -> List[re.Pattern]:
    s0 = s.strip()
    # уберём пробелы/nbsp в копии, чтобы отделить целую/десятичную часть
    s_compact = s0.replace(" ", "").replace(_SEP_NBSP, "").replace(_SEP_THIN, "")
    if re.fullmatch(r"\d+[.,]\d+", s_compact):
        int_part, frac_part = re.split(r"[.,]", s_compact, maxsplit=1)
        pat = _regex_for_float(int_part, frac_part)
        return [re.compile(pat)]
    elif re.fullmatch(r"\d+", s_compact):
        pat = _regex_for_int(s_compact)
        return [re.compile(pat)]
    # fallback: может быть "249.0" без смысла десятичной части
    m = re.fullmatch(r"(\d+)[.,]0+", s_compact)
    if m:
        pat = _regex_for_int(m.group(1))
        return [re.compile(pat)]
    return []

def _string_patterns_from_str(s: str) -> List[re.Pattern]:
    # нормализуем кавычки и пробелы, строим regex с гибкими пробелами и опц.кавычками
    s1 = _light_norm(s).strip('"').strip()
    # если строка выглядит как ФИО/название с кавычками — по токенам
    tokens = [re.escape(tok) for tok in re.split(r"\s+", s1) if tok]
    if not tokens:
        return []
    body = r"\s+".join(tokens)
    # допускаем отсутствие/наличие любых кавычек вокруг
    pat = rf"[\"“”«»]?\s*{body}\s*[\"“”«»]?"
    return [re.compile(pat, flags=re.IGNORECASE)]

class _RelaxedAnswerMatch:
    """
    Совместимый API:
      - compile_patterns(answer_str) -> List[Pattern]
      - any_match(text, patterns) -> (bool, matched_regex_as_str)
    """
    def compile_patterns(self, answer_str: Union[str, int, float]) -> List[re.Pattern]:
        s = _light_norm(str(answer_str))
        pats: List[re.Pattern] = []
        # сначала пытаемся как число
        if _is_numeric_string(s):
            pats.extend(_numeric_patterns_from_str(s))
        # потом как строка (на случай "АО \"...\"" и т.п.)
        pats.extend(_string_patterns_from_str(s))
        # удалим дубликаты по pattern/flags
        uniq = []
        seen = set()
        for p in pats:
            key = (p.pattern, p.flags)
            if key not in seen:
                seen.add(key)
                uniq.append(p)
        return uniq

    def any_match(self, text: str, patterns: Iterable[re.Pattern]) -> Tuple[bool, str]:
        if not text:
            return (False, "")
        # без агрессивной нормализации текста: числовые regex уже допускают разные разделители
        t = text.replace(_SEP_THIN, " ").replace(_SEP_NBSP, " ")
        for p in patterns:
            if p.search(t):
                return (True, p.pattern)
        return (False, "")

relaxed_answer_match = _RelaxedAnswerMatch()
