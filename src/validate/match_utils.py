# src/validate/match_utils.py
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, List, Tuple, Iterable


# --- Нормализация текста -----------------------------------------------------


def _norm(s: str) -> str:
    """
    Мягкая нормализация:
    - NFKC (приводит «странные» юникодные формы к обычным)
    - унификация кавычек и тире
    - замена неразрывных пробелов на обычные
    - схлопывание множественных пробелов
    """
    s = unicodedata.normalize("NFKC", str(s))

    # кавычки → "  / '
    s = (
        s.replace("«", '"')
        .replace("»", '"')
        .replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )

    # все виды тире → обычный дефис
    s = re.sub(r"[\u2010-\u2015]", "-", s)

    # NBSP / thin space → пробел
    s = s.replace("\u00A0", " ").replace("\u2009", " ")

    # схлопываем пробелы
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _mojibake_variants(s: str) -> List[str]:
    """Возвращает список возможных «кракозябренных» вариантов строки.

    Эвристика: пробуем перекодировать latin1/cp1252/cp1251 ↔ utf-8 и
    оставляем наиболее кириллические версии. Это помогает матчить тексты,
    где исходная строка была испорчена неверной декодировкой.
    """

    variants: set[str] = set()
    enc_pairs: Iterable[tuple[str, str]] = [
        ("latin1", "utf-8"),
        ("cp1252", "utf-8"),
        ("cp1251", "utf-8"),
        ("utf-8", "latin1"),
        ("utf-8", "cp1252"),
        ("utf-8", "cp1251"),
    ]
    for src_enc, dst_enc in enc_pairs:
        try:
            v = s.encode(src_enc, errors="ignore").decode(dst_enc, errors="ignore")
            variants.add(v)
        except Exception:
            pass

    variants.discard(s)

    def cyr_score(text: str) -> int:
        return len(re.findall(r"[А-Яа-яЁё]", text))

    # сортируем по количеству кириллицы, оставляем наиболее правдоподобные
    return sorted(variants, key=cyr_score, reverse=True)[:3]

# --- Построение паттернов ----------------------------------------------------


_GAP = r"[ \t\r\n\.,;:!?\-–—\"'()]{0,3}"


def _tokens_pattern(answer: str, gap: str = _GAP) -> str:
    """
    Паттерн по токенам с небольшим допуском на знаки/пробелы между ними.
    Используем как основу для текстовых ответов.
    """
    toks = re.findall(r"\w+|\d+[.,]\d+|\d+", answer, flags=re.UNICODE)
    if not toks:
        return re.escape(answer)
    return r"\b" + gap.join(map(re.escape, toks)) + r"\b"


@dataclass
class RelaxedAnswerMatch:
    """
    Помощник для «релаксированного» сопоставления ответа внутри текста чанков.
    """

    def compile_patterns(self, answer: Any) -> List[str]:
        """
        Вернуть список строк-паттернов (regex) для данного ответа.
        Поддерживает:
          - целые числа: 300, "23540"
          - десятичные: 6.2, "249,747" (разделитель ',' или '.')
          - текст: «мягкая» склейка токенов (кавычки/точки/дефисы допускаются между словами)
        """
        if answer is None:
            return []

        # --- ЧИСЛА -----------------------------------------------------------
        # распознаём число, даже если пользователь пришёл строкой
        as_str = str(answer).strip()
        as_str_norm = _norm(as_str)

        # чисто цифры (целое), допускаем разделители тысяч
        only_digits = re.fullmatch(r"\d+(?:\s+\d+)*", as_str_norm)
        # десятичное число с точкой или запятой и опциональными разделителями тысяч
        decimal_match = re.fullmatch(
            r"\d+(?:\s+\d+)*(?:\s*[.,]\s*\d+)", as_str_norm
        )

        if only_digits:
            digits = re.sub(r"\s+", "", as_str_norm)
            p1 = rf"(?<!\d){re.escape(digits)}(?!\d)"
            spaced = r"".join(
                [re.escape(d) + r"[\s\u00A0\u2009]*" for d in digits]
            )
            p2 = rf"(?<!\d){spaced}(?!\d)"
            return [p1, p2]

        if decimal_match:
            canon = re.sub(r"\s*", "", as_str_norm.replace(",", "."))
            head, tail = canon.split(".", 1)
            p1 = (
                rf"(?<!\d){re.escape(head)}[\s\u00A0\u2009]*[.,][\s\u00A0\u2009]*{re.escape(tail)}(?!\d)"
            )
            spaced_head = "".join(
                [re.escape(d) + r"[\s\u00A0\u2009]*" for d in head]
            )
            spaced_tail = "".join(
                [re.escape(d) + r"[\s\u00A0\u2009]*" for d in tail]
            )
            p2 = (
                rf"(?<!\d){spaced_head}[\s\u00A0\u2009]*[.,][\s\u00A0\u2009]*{spaced_tail}(?!\d)"
            )
            return [p1, p2]

        # --- ТЕКСТ -----------------------------------------------------------
        s = _norm(as_str)
        if not s:
            return []

        variants = [s] + _mojibake_variants(s)
        patterns: List[str] = []
        for v in variants:
            v_norm = _norm(v)
            patterns.append(_tokens_pattern(v_norm, gap=_GAP))
            patterns.append(_tokens_pattern(v_norm, gap=r"\W*"))

        # удалим дубли, сохраняя порядок
        seen = {}
        return [seen.setdefault(p, p) for p in patterns if p not in seen]

    def any_match(self, text: str, patterns: List[str]) -> Tuple[bool, str]:
        """
        Проверить, совпадает ли `text` с любым из паттернов.
        Возвращает (True/False, сработавший_паттерн_или_пусто).
        """
        t = _norm(text)
        for p in patterns:
            if re.search(p, t, flags=re.IGNORECASE | re.UNICODE | re.DOTALL):
                return True, p
        return False, ""


# Экспортируем объект-namespace с методами compile_patterns / any_match
relaxed_answer_match = RelaxedAnswerMatch()

__all__ = ["relaxed_answer_match", "RelaxedAnswerMatch"]
