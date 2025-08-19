# src/validate/match_utils.py
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, List, Tuple


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


# --- Построение паттернов ----------------------------------------------------


def _tokens_pattern(answer: str) -> str:
    """
    «Склеивающий» паттерн по токенам: разрешаем любые не-алфавитно-цифровые
    символы между токенами. Это ловит варианты вроде `L.A.C. Holding`,
    `L A C Holding`, кавычки/скобки/дефисы между словами и т.п.
    """
    # целые/десятичные числа выделим отдельно в compile_patterns()
    # здесь нам нужны слова/буквы/цифры как токены
    toks = re.findall(r"\w+|\d+[.,]\d+|\d+", answer, flags=re.UNICODE)
    if not toks:
        return re.escape(answer)

    # \W* между токенами: любые не-словарные символы (точки, пробелы, кавычки...)
    # границы слова по краям помогают избегать «прилипаний» к соседним словам
    return r"\b" + r"\W*".join(map(re.escape, toks)) + r"\b"


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

        # чисто цифры (целое), допускаем, что внутри могли быть пробелы как разделители тысяч
        only_digits = re.fullmatch(r"\d+(?:\s+\d+)*", as_str_norm)
        # десятичное с точкой/запятой (возможны пробелы вокруг разделителя)
        decimal_match = re.fullmatch(r"\d+(?:\s*[.,]\s*\d+)", as_str_norm)

        if only_digits:
            digits = re.sub(r"\s+", "", as_str_norm)  # убираем разделители тысяч
            # 1) точное число (без прилегающих цифр)
            p1 = rf"(?<!\d){re.escape(digits)}(?!\d)"
            # 2) тот же набор цифр, но с допуском пробелов/NBSP/узкого пробела между ЛЮБЫМИ цифрами
            spaced = r"".join([re.escape(d) + r"[\s\u00A0\u2009]*" for d in digits])
            p2 = rf"(?<!\d){spaced}(?!\d)"
            return [p1, p2]

        if decimal_match:
            # приводим к точке как «каноническому» варианту
            canon = re.sub(r"\s*", "", as_str_norm.replace(",", "."))
            head, tail = canon.split(".", 1)
            # допускаем пробелы вокруг разделителя и сам разделитель , или .
            p = rf"(?<!\d){re.escape(head)}[\s\u00A0\u2009]*[.,][\s\u00A0\u2009]*{re.escape(tail)}(?!\d)"
            return [p]

        # --- ТЕКСТ -----------------------------------------------------------
        s = _norm(as_str)
        if not s:
            return []

        # основной «склеивающий» паттерн по токенам
        p_main = _tokens_pattern(s)

        # запасной вариант: чуть строже, только ограниченный набор знаков между словами
        toks = [re.escape(t) for t in re.split(r"\s+", s)]
        gap_lite = r"[ \t\r\n\.,;:!?\-–—\"'()]{0,3}"
        p_lite = gap_lite.join(toks)

        return [p_main, p_lite]

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
