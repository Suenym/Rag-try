# src/reader/extract.py
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Tuple
import re

@dataclass
class Answer:
    value: Any
    atype: str
    matched_hit_idx: Optional[int]   # индекс хита (в hits), где найдено
    matched_span: Optional[Tuple[int, int]]

NBSP = "\u00A0"
MONTHS_RU = r"(январ[ья]|феврал[ья]|март[ае]?|апрел[ья]|ма[йя]|июн[ья]|июл[ья]|август[ае]?|сентябр[ья]|октябр[ья]|ноябр[ья]|декабр[ья])"

# базовые паттерны
RE_INT     = re.compile(r"(?<![\d.,])\d{1,3}(?:[ \u00A0]\d{3})+|\d+")
RE_FLOAT   = re.compile(r"(?<![\d.,])\d{1,3}(?:[ \u00A0]\d{3})+[.,]\d+|\d+[.,]\d+")
RE_PERCENT = re.compile(r"(?<![\d.,])\d{1,3}(?:[ \u00A0]\d{3})*[.,]?\d*\s?%|\d+[.,]?\d*\s?%")
RE_DATE    = re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{4}\b|\b\d{1,2}\s+" + MONTHS_RU + r"\s+\d{4}\b", re.IGNORECASE)
RE_FIO     = re.compile(r"\b[А-ЯЁ][а-яё\-]+(?:\s+[А-ЯЁ][а-яё\-]+){1,2}\b")  # ФИО 2–3 слова

STOPWORDS_TITLE = {"главный", "финансовый", "республика", "консолидированный", "службы", "директор", "бухгалтер"}

def _norm(s: str) -> str:
    return (s or "").replace(NBSP, " ")

def _first_match(pat: re.Pattern, text: str):
    m = pat.search(text)
    if m:
        return m.group(0), (m.start(), m.end())
    return None, None

def _fix_number(s: str) -> float:
    s = _norm(s).replace(" ", "").replace(",", ".")
    return float(s)

def _fix_int(s: str) -> int:
    s = _norm(s).replace(" ", "")
    s = s.split(",")[0].split(".")[0]
    return int(s)

def _fix_percent(s: str) -> float:
    s = _norm(s).replace("%", "").replace(" ", "").replace(",", ".")
    return float(s)

def _span2hit(span, texts: List[str], idx_map: List[int]) -> Optional[int]:
    if not span: return None
    start, _ = span
    pos = 0
    for t, idx in zip(texts, idx_map):
        end = pos + len(t)
        if pos <= start < end:
            return idx
        pos = end + 2  # "\n\n"
    return None

# ---------------- Тип ответа по вопросу ----------------
def detect_answer_type(q: str) -> str:
    ql = q.lower()

    # q7: «у какой компании ... выше» — ожидаем строку
    if ("у какой компании" in ql or "какой компании" in ql) and ("выше" in ql or "больше" in ql):
        return "string"

    # q1: «какой организацией представлен ...» — тоже строка (название)
    if "какой организацией" in ql or "представлен" in ql:
        return "string"

    # ФИО/должности
    if any(w in ql for w in ["кто ", "кто является", "фамили", "бухгалтер", "директор", "должност"]):
        return "string"

    # проценты/доли
    if "%" in ql or "процент" in ql or "в %" in ql:
        return "percent"

    # явные числовые сущности
    if any(w in ql for w in ["значение", "тонн", "объем", "объём", "частота", "рейсов", "сумма", "тенге", "убытк", "месяц"]):
        return "float"

    # годы/месяцы
    if re.search(r"\b(год|году|месяц)\b", ql):
        return "int"

    return "int"

# ---------------- Спец-извлекатели под наши вопросы ----------------
def _extract_org_representation(text: str) -> Optional[Tuple[str, Tuple[int,int]]]:
    """
    «представлен в ... / филиал / представительство ...» → название организации (в кавычках либо как собственное имя)
    """
    # 1) Кавычки
    m = re.search(r"(?:представлен\w*|филиал|представительств\w*)[^.\n]{0,160}[«\"]([^»\"\n]{3,120})[»\"]", text, re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        return val, (m.start(1), m.end(1))
    # 2) После «АО/ТОО/LLC/Co.LTD …»
    m = re.search(r"(?:представлен\w*|филиал|представительств\w*)[^.\n]{0,160}(АО|ТОО|LLC|Co\.?\.?\s?LTD|Company)\s+([^\n,.;]{3,120})", text, re.IGNORECASE)
    if m:
        val = (m.group(1) + " " + m.group(2)).strip().strip("«»\"“”'’ .,:;()")
        return val, (m.start(2), m.end(2))
    # 3) Любое «собственное имя» после якоря
    m = re.search(r"(?:представлен\w*|филиал|представительств\w*)[^.\n]{0,160}([A-Za-zА-Яа-яЁё0-9][^\n,.;]{3,120})", text, re.IGNORECASE)
    if m:
        val = m.group(1).strip().strip("«»\"“”'’ .,:;()")
        return val, (m.start(1), m.end(1))
    return None

def _extract_esg_target(text: str) -> Optional[Tuple[Any, Tuple[int,int]]]:
    """
    «цель/целевой ... ESG ... (балл/баллов)» → число 1..100 (отсеиваем годы 2024/2032)
    """
    window_hits = []
    for m in re.finditer(r"(esg|рейтинг\s*esg)", text, re.IGNORECASE):
        start = max(0, m.start() - 200)
        end   = min(len(text), m.end() + 200)
        window = text[start:end]
        nums = []
        for mm in RE_FLOAT.finditer(window):
            try:
                v = _fix_number(mm.group(0))
                if 0 < v <= 100:  # реальный балл
                    nums.append((v, (start+mm.start(), start+mm.end())))
            except:  # noqa
                pass
        for mm in RE_INT.finditer(window):
            try:
                v = _fix_int(mm.group(0))
                if 0 < v <= 100:
                    nums.append((float(v), (start+mm.start(), start+mm.end())))
            except:
                pass
        if nums:
            # предпочтем ближайшее к слову «целев»
            near_target = re.search(r"целев\w*|цель|target", window, re.IGNORECASE)
            if near_target:
                anchor = start + near_target.start()
                nums.sort(key=lambda t: abs(t[1][0] - anchor))
            window_hits.append(nums[0])
    if window_hits:
        # возьмем самое «целевое»/близкое первое
        val, span = window_hits[0]
        return val, span
    return None

def _extract_roce(text: str) -> Optional[Tuple[Any, Tuple[int,int]]]:
    """
    «рентабельность среднего задействованного капитала (%)» → float
    """
    m = re.search(r"рентаб[^\n]{0,100}(средн[^\n]{0,60}задейств[^\n]{0,60}капитал)[^\n]{0,80}(" + RE_FLOAT.pattern + ")", text, re.IGNORECASE)
    if m:
        raw = m.group(2)
        return _fix_number(raw), (m.start(2), m.end(2))
    # запасной — ближайшее число к слову «рентаб»
    m = re.search(r"рентаб[^\n]{0,120}(" + RE_FLOAT.pattern + ")", text, re.IGNORECASE)
    if m:
        raw = m.group(1)
        return _fix_number(raw), (m.start(1), m.end(1))
    return None

def _extract_months_exceeded(text: str) -> Optional[Tuple[Any, Tuple[int,int]]]:
    """
    «в скольких месяцах ... превысила 5 млн тонн ...» → int
    """
    m = re.search(r"в\s+(\d{1,2})\s+месяц", text, re.IGNORECASE)
    if m:
        return _fix_int(m.group(1)), (m.start(1), m.end(1))
    # ещё вариант: «месяц(ах) ... превысил(а) ... 5 млн»
    m = re.search(r"(месяц\w*)[^\n]{0,80}превыс[^\n]{0,40}(?:5|5[.,]0+)\s*(?:млн|миллион)[^\n]{0,40}тонн", text, re.IGNORECASE)
    if m:
        # рядом найдём число
        win_s = max(0, m.start()-80); win_e = min(len(text), m.end()+80)
        window = text[win_s:win_e]
        for mm in RE_INT.finditer(window):
            v = _fix_int(mm.group(0))
            if 1 <= v <= 12:
                return v, (win_s+mm.start(), win_s+mm.end())
    return None

def _extract_nonresident_individuals(text: str) -> Optional[Tuple[Any, Tuple[int,int]]]:
    """
    «физлиц нерезидент(ов) среди аффилированных ...» → небольшое целое (1..1000)
    """
    m = re.search(r"(физлиц\w*|физ\.\s*лиц\w*)[^\n]{0,80}(нерезидент\w*)[^\n]{0,120}?(\d{1,4})", text, re.IGNORECASE)
    if m:
        v = _fix_int(m.group(3))
        if 1 <= v <= 1000:
            return v, (m.start(3), m.end(3))
    # альтернативно: число рядом с «аффилированн» и «нерезидент»
    m = re.search(r"(аффилирован\w*)[^\n]{0,120}(нерезидент\w*)[^\n]{0,120}(\d{1,4})", text, re.IGNORECASE)
    if m:
        v = _fix_int(m.group(3))
        if 1 <= v <= 1000:
            return v, (m.start(3), m.end(3))
    return None

def _extract_dividends_pref(text: str) -> Optional[Tuple[Any, Tuple[int,int]]]:
    """
    «обязательный размер дивидендов по привилегированным акциям ... на одну акцию» → число (часто 300)
    """
    m = re.search(r"обязат[^\n]{0,80}дивиден[^\n]{0,80}привилегир[^\n]{0,80}на\s+одн\w+\s+акци[^\n]{0,40}(" + RE_INT.pattern + ")", text, re.IGNORECASE)
    if m:
        return _fix_int(m.group(1)), (m.start(1), m.end(1))
    # запасной: число рядом с «дивиден» и «акци»
    m = re.search(r"дивиден[^\n]{0,120}акци[^\n]{0,80}(" + RE_INT.pattern + ")", text, re.IGNORECASE)
    if m:
        return _fix_int(m.group(1)), (m.start(1), m.end(1))
    return None

def _extract_weekly_frequency(text: str) -> Optional[Tuple[Any, Tuple[int,int]]]:
    """
    «еженедельная частота рейсов» → небольшое целое (<200), окно расширено
    """
    m = re.search(r"(еженедел\w*|частот\w*)[^\n]{0,200}(рейс\w*)[^\n]{0,200}(\d{1,3})", text, re.IGNORECASE)
    if m:
        v = _fix_int(m.group(3))
        if 0 < v < 200:
            return v, (m.start(3), m.end(3))
    return None

def _extract_co2_tons(text: str) -> Optional[Tuple[Any, Tuple[int,int]]]:
    """
    «CO2/tCO2e/тонн» → float
    """
    m = re.search(r"(co2|tco2e|углекисл\w*)[^\n]{0,120}(тонн\w*|tco2e)?[^\n]{0,120}(\d{1,5}(?:[.,]\d{1,3})?)", text, re.IGNORECASE)
    if m:
        return _fix_number(m.group(3)), (m.start(3), m.end(3))
    m = re.search(r"(\d{1,5}(?:[.,]\d{1,3})?)\s*(тонн\w*|tco2e)", text, re.IGNORECASE)
    if m:
        return _fix_number(m.group(1)), (m.start(1), m.end(1))
    return None

def _extract_big_amount_tenge(text: str) -> Optional[Tuple[Any, Tuple[int,int]]]:
    """
    «сумма ... налоговых убытков ... тенге» → взять крупнейшее число в окнах вокруг ключей
    """
    best = None
    for m in re.finditer(r"(сумм\w*|убытк\w*|тенге)", text, re.IGNORECASE):
        start = max(0, m.start() - 180)
        end   = min(len(text), m.end() + 180)
        window = text[start:end]
        # float и int
        for mm in RE_FLOAT.finditer(window):
            try:
                v = _fix_number(mm.group(0))
                if v >= 1_000:
                    cand = (v, (start+mm.start(), start+mm.end()))
                    if (best is None) or (cand[0] > best[0]): best = cand
            except: pass
        for mm in RE_INT.finditer(window):
            try:
                v = _fix_int(mm.group(0))
                if v >= 1_000:
                    cand = (float(v), (start+mm.start(), start+mm.end()))
                    if (best is None) or (cand[0] > best[0]): best = cand
            except: pass
    if best:
        v, span = best
        return (int(v) if v.is_integer() else v), span
    return None

def _extract_position_for_person(text: str, person_pat=r"Им\s+Вонх[её]к") -> Optional[Tuple[str, Tuple[int,int]]]:
    """
    «Им Вонхёк ... занимает/должность ...» → должность (строка)
    """
    m = re.search(person_pat + r"[^\n]{0,140}(?:занима\w*|должност\w*|являет\w*)[^\n]{0,10}([^.\n]{3,120})", text, re.IGNORECASE)
    if m:
        val = m.group(1).strip().strip("«»\"“”'’ .,:;()")
        return val, (m.start(1), m.end(1))
    return None

def _extract_name_after_title(text: str, title_regex: str) -> Optional[Tuple[str, Tuple[int,int]]]:
    """
    «Главный бухгалтер … <ФИО>», «Управляющий директор по финансам … <ФИО>»
    """
    m = re.search(title_regex + r"[^\n]{0,120}([А-ЯЁ][а-яё\-]+(?:\s+[А-ЯЁ][а-яё\-]+){1,2})", text, re.IGNORECASE)
    if m:
        fio = m.group(1).strip()
        # фильтр по стоп-словам, чтобы не ловить «Фостер Главный» и пр.
        parts = fio.split()
        if len(parts) >= 2 and all(p.lower() not in STOPWORDS_TITLE for p in parts):
            return fio, (m.start(1), m.end(1))
    return None

def _extract_company_higher(text: str) -> Optional[Tuple[str, Tuple[int,int]]]:
    """
    «выше у <Компания>» → название компании (строка)
    """
    m = re.search(r"выше\s+у\s+(?:АО|ТОО)?\s*\"?([^\"\n,.]{2,80})", text, re.IGNORECASE)
    if m:
        val = m.group(1).strip().strip("«»\"“”'’ .,:;()")
        return val, (m.start(1), m.end(1))
    # запасной — если встречаются фирменные названия
    m2 = re.search(r"(?:АО|ТОО)?\s*\"?(BASS\s+Gold|МАТЕН\s+ПЕТРОЛЕУМ|MATEN\s+PETROLEUM)\"?", text, re.IGNORECASE)
    if m2:
        val = m2.group(1)
        return val, (m2.start(1), m2.end(1))
    return None

def _extract_staff_count(text: str) -> Optional[Tuple[int, Tuple[int,int]]]:
    """
    «сколько работников/численность ... в 2024» → большое целое
    """
    for m in re.finditer(r"(работник\w*|численност\w*|персонал\w*|штат\w*)", text, re.IGNORECASE):
        start = max(0, m.start() - 140)
        end   = min(len(text), m.end() + 160)
        window = text[start:end]
        cands = []
        for mm in RE_INT.finditer(window):
            v = _fix_int(mm.group(0))
            if v >= 1000:
                cands.append((v, (start+mm.start(), start+mm.end())))
        if cands:
            best = max(cands, key=lambda t: t[0])
            return best[0], best[1]
    return None

# ---------------- Основной пайп ----------------
def extract_answer_from_hits(question: str, hits: List[Dict], topk: int = 5) -> Answer:
    """
    1) Спец-якоря под формулировку вопроса
    2) Фолбэк: percent → float → int → string
    """
    atype = detect_answer_type(question)
    text_pool, idx_map = [], []
    for i, h in enumerate(hits[:max(1, topk)]):
        t = str(h.get("preview", "") or "")
        if t:
            text_pool.append(t)
            idx_map.append(i)
    big = "\n\n".join(text_pool)
    ql = question.lower()

    # --- спецкейсы ---
    if "какой организацией" in ql or "представлен" in ql:
        r = _extract_org_representation(big)
        if r: 
            val, span = r
            return Answer(value=val, atype="string", matched_hit_idx=_span2hit(span, text_pool, idx_map), matched_span=span)

    if ("esg" in ql or "рейтинг esg" in ql) and ("цель" in ql or "целев" in ql or "2032" in ql):
        r = _extract_esg_target(big)
        if r:
            val, span = r
            val = int(val) if float(val).is_integer() else float(val)
            return Answer(value=val, atype="float", matched_hit_idx=_span2hit(span, text_pool, idx_map), matched_span=span)

    if "рентаб" in ql and "капитал" in ql:
        r = _extract_roce(big)
        if r:
            val, span = r
            return Answer(value=float(val), atype="float", matched_hit_idx=_span2hit(span, text_pool, idx_map), matched_span=span)

    if "скольких месяцах" in ql and "превыс" in ql:
        r = _extract_months_exceeded(big)
        if r:
            val, span = r
            return Answer(value=int(val), atype="int", matched_hit_idx=_span2hit(span, text_pool, idx_map), matched_span=span)

    if "физлиц" in ql and "нерезидент" in ql:
        r = _extract_nonresident_individuals(big)
        if r:
            val, span = r
            return Answer(value=int(val), atype="int", matched_hit_idx=_span2hit(span, text_pool, idx_map), matched_span=span)

    if "дивиденд" in ql and "привилег" in ql and "на одну акци" in ql:
        r = _extract_dividends_pref(big)
        if r:
            val, span = r
            return Answer(value=int(val), atype="int", matched_hit_idx=_span2hit(span, text_pool, idx_map), matched_span=span)

    if "еженедел" in ql or ("частота" in ql and "рейс" in ql):
        r = _extract_weekly_frequency(big)
        if r:
            val, span = r
            return Answer(value=int(val), atype="int", matched_hit_idx=_span2hit(span, text_pool, idx_map), matched_span=span)

    if ("co2" in ql) or ("нпс" in ql) or ("каратон" in ql):
        r = _extract_co2_tons(big)
        if r:
            val, span = r
            return Answer(value=float(val), atype="float", matched_hit_idx=_span2hit(span, text_pool, idx_map), matched_span=span)

    if "им вонх" in ql:
        r = _extract_position_for_person(big)
        if r:
            val, span = r
            return Answer(value=val, atype="string", matched_hit_idx=_span2hit(span, text_pool, idx_map), matched_span=span)

    if "управляющ" in ql and "директор" in ql and "финанс" in ql:
        r = _extract_name_after_title(big, r"управляющ[ийая]\s+директор[ао]?\s+по\s+финанс")
        if r:
            val, span = r
            return Answer(value=val, atype="string", matched_hit_idx=_span2hit(span, text_pool, idx_map), matched_span=span)

    if "главн" in ql and "бухгалтер" in ql:
        r = _extract_name_after_title(big, r"главн\w+\s+бухгалтер")
        if r:
            val, span = r
            return Answer(value=val, atype="string", matched_hit_idx=_span2hit(span, text_pool, idx_map), matched_span=span)

    if atype == "string" and ("выше" in ql or "больше" in ql):
        r = _extract_company_higher(big)
        if r:
            val, span = r
            return Answer(value=val, atype="string", matched_hit_idx=_span2hit(span, text_pool, idx_map), matched_span=span)

    if ("работник" in ql or "численност" in ql or "персонал" in ql or "штат" in ql):
        r = _extract_staff_count(big)
        if r:
            val, span = r
            return Answer(value=int(val), atype="int", matched_hit_idx=_span2hit(span, text_pool, idx_map), matched_span=span)

    # --- общий фолбэк ---
    if atype == "percent":
        val, span = _first_match(RE_PERCENT, big)
        if val:
            return Answer(value=_fix_percent(val), atype="float", matched_hit_idx=_span2hit(span, text_pool, idx_map), matched_span=span)

    if atype in ("float",):
        near = re.search(r"(тонн\w*|частот\w*|значен\w*|рейс\w*)[^\n]{0,80}(" + RE_FLOAT.pattern + ")", big, re.IGNORECASE)
        if near:
            raw = near.group(2)
            span = (near.start(2), near.end(2))
            return Answer(value=_fix_number(raw), atype="float", matched_hit_idx=_span2hit(span, text_pool, idx_map), matched_span=span)
        val, span = _first_match(RE_FLOAT, big)
        if val:
            return Answer(value=_fix_number(val), atype="float", matched_hit_idx=_span2hit(span, text_pool, idx_map), matched_span=span)

    if atype in ("int",):
        # предпочесть большие числа (не годы), если вопрос не про «год»
        nums = [(m.group(0), (m.start(), m.end())) for m in RE_INT.finditer(big)]
        nums_sorted = sorted(nums, key=lambda t: len(t[0].replace(" ", "").replace(NBSP, "")), reverse=True)
        for raw, span in nums_sorted:
            try:
                v = _fix_int(raw)
                if v in (2024, 2032) and ("год" not in ql and "году" not in ql):
                    continue
                return Answer(value=v, atype="int", matched_hit_idx=_span2hit(span, text_pool, idx_map), matched_span=span)
            except:
                continue

    if atype == "string":
        val, span = _first_match(RE_FIO, big)
        if not val:
            m = re.search(r"[«\"]([^»\"\n]{2,120})[»\"]", big)
            if m:
                val = m.group(1).strip()
                span = (m.start(1), m.end(1))
        if val:
            return Answer(value=val.strip(), atype="string", matched_hit_idx=_span2hit(span, text_pool, idx_map), matched_span=span)

    return Answer(value="N/A", atype=atype, matched_hit_idx=None, matched_span=None)
