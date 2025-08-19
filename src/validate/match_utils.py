# src/validate/match_utils.py
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, List, Tuple, Iterable

# ——— Нормализация текста ————————————————————————————————————————————————

def _norm(s: str) -> str:
    """
    Мягкая нормализация:
    - NFKC (приведение “странных” форм к обычным)
    - унификация кавычек/тире
    - NBSP/thin space → обычный пробел
    - схлопывание пробелов
    """
    s = unicodedata.normalize("NFKC", str(s))

    # кавычки → " / '
    s = (
        s.replace("«", '"').replace("»", '"')
         .replace("“", '"').replace("”", '"')
         .replace("‘", "'").replace("’", "'")
    )

    # все виды тире → дефис
    s = re.sub(r"[\u2010-\u2015]", "-", s)

    # NBSP / thin space → пробел
    s = s.replace("\u00A0", " ").replace("\u2009", " ")

    # схлопываем пробелы
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _mojibake_variants(s: str) -> List[str]:
    """
    Возвращает 0–3 «кракозябренных» вариантов строки (эвристика),
    чтобы матчить тексты, испорченные неверной декодировкой.
    """
    variants: set[str] = set()
    enc_pairs: Iterable[tuple[str, str]] = [
        ("latin1", "utf-8"), ("cp1252", "utf-8"), ("cp1251", "utf-8"),
        ("utf-8", "latin1"), ("utf-8", "cp1252"), ("utf-8", "cp1251"),
    ]
    for src_enc, dst_enc in enc_pairs:
        try:
            v = s.encode(src_enc, errors="ignore").decode(dst_enc, errors="ignore")
            if v and v != s:
                variants.add(v)
        except Exception:
            pass

    def cyr_score(text: str) -> int:
        return len(re.findall(r"[А-Яа-яЁё]", text))

    return sorted(variants, key=cyr_score, reverse=True)[:3]

# ——— Построение паттернов ————————————————————————————————————————————————

_GAP = r"[ \t\r\n\.,;:!?\-–—\"'()]{0,3}"

def _tokens_pattern(answer: str, gap: str = _GAP) -> str:
    """
    Паттерн по токенам с небольшим допуском знаков/пробелов между ними.
    Используем как основу для текстовых ответов.
    """
    toks = re.findall(r"\w+|\d+[.,]\d+|\d+", answer, flags=re.UNICODE)
    if not toks:
        return re.escape(answer)
    return gap.join(map(re.escape, toks))

@dataclass
class RelaxedAnswerMatch:
    """Релаксированное сопоставление ответа внутри текста чанков."""

    def compile_patterns(self, answer: Any) -> List[str]:
        """
        Вернуть список regex-паттернов для ответа.
        Поддерживаем:
          - целые: 300, "23540" (+ вариант 3\s*0\s*0)
          - десятичные: 6.2 / "6,2" (разделитель . или , и «люфт» пробелов)
          - текст: склейка токенов с допустимыми разделителями
          - mojibake-варианты текста
        """
        if answer is None:
            return []

        as_str = str(answer).strip()
        as_norm = _norm(as_str)

        # — ЧИСЛА ——————————————————————————————————————————
        # целое (с возможными разделителями тысяч пробелами/NBSP/thin space)
        if re.fullmatch(r"\d+(?:[ \u00A0\u2009]\d+)*", as_norm):
            digits = re.sub(r"[ \u00A0\u2009]+", "", as_norm)
            # точное без прилегающих цифр
            p1 = rf"(?<!\d){re.escape(digits)}(?!\d)"
            # «разрежённый» вариант: 3\s*0\s*0
            spaced = r"".join([re.escape(d) + r"[\s\u00A0\u2009]*" for d in digits])
            p2 = rf"(?<!\d){spaced}(?!\d)"
            return [p1, p2]

        # десятичное: допускаем , или . с «люфтом», и разделители тысяч в целой части
        if re.fullmatch(r"\d+(?:[ \u00A0\u2009]\d+)*[ ,.]\s*\d+", as_norm):
            canon = as_norm.replace(",", ".")
            head = re.sub(r"[ \u00A0\u2009]+", "", canon).split(".", 1)[0]
            tail = canon.split(".", 1)[1].strip()
            # плотный вариант
            p1 = rf"(?<!\d){re.escape(head)}[\s\u00A0\u2009]*[.,][\s\u00A0\u2009]*{re.escape(tail)}(?!\d)"
            # «разрежённая» целая и дробная части
            spaced_head = "".join([re.escape(d) + r"[\s\u00A0\u2009]*" for d in head])
            spaced_tail = "".join([re.escape(d) + r"[\s\u00A0\u2009]*" for d in tail])
            p2 = rf"(?<!\d){spaced_head}[\s\u00A0\u2009]*[.,][\s\u00A0\u2009]*{spaced_tail}(?!\d)"
            return [p1, p2]

        # — ТЕКСТ ——————————————————————————————————————————
        s = as_norm
        if not s:
            return []
        base_patterns: List[str] = []

        # основной и «широкий» зазор
        base_patterns.append(_tokens_pattern(s, gap=_GAP))
        base_patterns.append(_tokens_pattern(s, gap=r"\W*"))

        # mojibake-варианты
        for v in _mojibake_variants(s):
            v_norm = _norm(v)
            base_patterns.append(_tokens_pattern(v_norm, gap=_GAP))
            base_patterns.append(_tokens_pattern(v_norm, gap=r"\W*"))

        # удалим дубликаты, сохраняя порядок
        seen = set()
        uniq = []
        for p in base_patterns:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return uniq

    def any_match(self, text: str, patterns: List[str]) -> Tuple[bool, str]:
        """Проверяем, совпадает ли текст с любым паттерном."""
        t = _norm(text)
        for pat in patterns:
            if re.search(pat, t, flags=re.IGNORECASE | re.UNICODE | re.DOTALL):
                return True, pat
        return False, ""

# Публичный namespace-объект
relaxed_answer_match = RelaxedAnswerMatch()

__all__ = ["relaxed_answer_match", "RelaxedAnswerMatch"]
