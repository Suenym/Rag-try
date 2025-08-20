# src/validate/match_utils.py
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, List, Tuple, Iterable

# â€”â€”â€” ÐÐ¾Ñ€Ð¼Ð°Ð»Ð¸Ð·Ð°Ñ†Ð¸Ñ Ñ‚ÐµÐºÑÑ‚Ð° â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”

def _norm(s: str) -> str:
    """
    ÐœÑÐ³ÐºÐ°Ñ Ð½Ð¾Ñ€Ð¼Ð°Ð»Ð¸Ð·Ð°Ñ†Ð¸Ñ:
    - NFKC (Ð¿Ñ€Ð¸Ð²ÐµÐ´ÐµÐ½Ð¸Ðµ â€œÑÑ‚Ñ€Ð°Ð½Ð½Ñ‹Ñ…â€ Ñ„Ð¾Ñ€Ð¼ Ðº Ð¾Ð±Ñ‹Ñ‡Ð½Ñ‹Ð¼)
    - ÑƒÐ½Ð¸Ñ„Ð¸ÐºÐ°Ñ†Ð¸Ñ ÐºÐ°Ð²Ñ‹Ñ‡ÐµÐº/Ñ‚Ð¸Ñ€Ðµ
    - NBSP/thin space â†’ Ð¾Ð±Ñ‹Ñ‡Ð½Ñ‹Ð¹ Ð¿Ñ€Ð¾Ð±ÐµÐ»
    - ÑÑ…Ð»Ð¾Ð¿Ñ‹Ð²Ð°Ð½Ð¸Ðµ Ð¿Ñ€Ð¾Ð±ÐµÐ»Ð¾Ð²
    """
    s = unicodedata.normalize("NFKC", str(s))

    # ÐºÐ°Ð²Ñ‹Ñ‡ÐºÐ¸ â†’ " / '
    s = (
        s.replace("Â«", '"').replace("Â»", '"')
         .replace("â€œ", '"').replace("â€", '"')
         .replace("â€˜", "'").replace("â€™", "'")
    )

    # Ð²ÑÐµ Ð²Ð¸Ð´Ñ‹ Ñ‚Ð¸Ñ€Ðµ â†’ Ð´ÐµÑ„Ð¸Ñ
    s = re.sub(r"[\u2010-\u2015]", "-", s)

    # NBSP / thin space â†’ Ð¿Ñ€Ð¾Ð±ÐµÐ»
    s = s.replace("\u00A0", " ").replace("\u2009", " ")

    # ÑÑ…Ð»Ð¾Ð¿Ñ‹Ð²Ð°ÐµÐ¼ Ð¿Ñ€Ð¾Ð±ÐµÐ»Ñ‹
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _mojibake_variants(s: str) -> List[str]:
    """
    Ð’Ð¾Ð·Ð²Ñ€Ð°Ñ‰Ð°ÐµÑ‚ 0â€“3 Â«ÐºÑ€Ð°ÐºÐ¾Ð·ÑÐ±Ñ€ÐµÐ½Ð½Ñ‹Ñ…Â» Ð²Ð°Ñ€Ð¸Ð°Ð½Ñ‚Ð¾Ð² ÑÑ‚Ñ€Ð¾ÐºÐ¸ (ÑÐ²Ñ€Ð¸ÑÑ‚Ð¸ÐºÐ°),
    Ñ‡Ñ‚Ð¾Ð±Ñ‹ Ð¼Ð°Ñ‚Ñ‡Ð¸Ñ‚ÑŒ Ñ‚ÐµÐºÑÑ‚Ñ‹, Ð¸ÑÐ¿Ð¾Ñ€Ñ‡ÐµÐ½Ð½Ñ‹Ðµ Ð½ÐµÐ²ÐµÑ€Ð½Ð¾Ð¹ Ð´ÐµÐºÐ¾Ð´Ð¸Ñ€Ð¾Ð²ÐºÐ¾Ð¹.
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
        return len(re.findall(r"[Ð-Ð¯Ð°-ÑÐÑ‘]", text))

    return sorted(variants, key=cyr_score, reverse=True)[:3]

# â€”â€”â€” ÐŸÐ¾ÑÑ‚Ñ€Ð¾ÐµÐ½Ð¸Ðµ Ð¿Ð°Ñ‚Ñ‚ÐµÑ€Ð½Ð¾Ð² â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”

_GAP = r"[ \t\r\n\.,;:!?\-â€“â€”\"'()]{0,3}"

def _tokens_pattern(answer: str, gap: str = _GAP) -> str:
    """
    ÐŸÐ°Ñ‚Ñ‚ÐµÑ€Ð½ Ð¿Ð¾ Ñ‚Ð¾ÐºÐµÐ½Ð°Ð¼ Ñ Ð½ÐµÐ±Ð¾Ð»ÑŒÑˆÐ¸Ð¼ Ð´Ð¾Ð¿ÑƒÑÐºÐ¾Ð¼ Ð·Ð½Ð°ÐºÐ¾Ð²/Ð¿Ñ€Ð¾Ð±ÐµÐ»Ð¾Ð² Ð¼ÐµÐ¶Ð´Ñƒ Ð½Ð¸Ð¼Ð¸.
    Ð˜ÑÐ¿Ð¾Ð»ÑŒÐ·ÑƒÐµÐ¼ ÐºÐ°Ðº Ð¾ÑÐ½Ð¾Ð²Ñƒ Ð´Ð»Ñ Ñ‚ÐµÐºÑÑ‚Ð¾Ð²Ñ‹Ñ… Ð¾Ñ‚Ð²ÐµÑ‚Ð¾Ð².
    """
    toks = re.findall(r"\w+|\d+[.,]\d+|\d+", answer, flags=re.UNICODE)
    if not toks:
        return re.escape(answer)
    return gap.join(map(re.escape, toks))

@dataclass
class RelaxedAnswerMatch:
    """Ð ÐµÐ»Ð°ÐºÑÐ¸Ñ€Ð¾Ð²Ð°Ð½Ð½Ð¾Ðµ ÑÐ¾Ð¿Ð¾ÑÑ‚Ð°Ð²Ð»ÐµÐ½Ð¸Ðµ Ð¾Ñ‚Ð²ÐµÑ‚Ð° Ð²Ð½ÑƒÑ‚Ñ€Ð¸ Ñ‚ÐµÐºÑÑ‚Ð° Ñ‡Ð°Ð½ÐºÐ¾Ð²."""

    def compile_patterns(self, answer: Any) -> List[str]:
        """
        Ð’ÐµÑ€Ð½ÑƒÑ‚ÑŒ ÑÐ¿Ð¸ÑÐ¾Ðº regex-Ð¿Ð°Ñ‚Ñ‚ÐµÑ€Ð½Ð¾Ð² Ð´Ð»Ñ Ð¾Ñ‚Ð²ÐµÑ‚Ð°.
        ÐŸÐ¾Ð´Ð´ÐµÑ€Ð¶Ð¸Ð²Ð°ÐµÐ¼:
          - Ñ†ÐµÐ»Ñ‹Ðµ: 300, "23540" (+ Ð²Ð°Ñ€Ð¸Ð°Ð½Ñ‚ 3\\s*0\\s*0)
          - Ð´ÐµÑÑÑ‚Ð¸Ñ‡Ð½Ñ‹Ðµ: 6.2 / "6,2" (Ñ€Ð°Ð·Ð´ÐµÐ»Ð¸Ñ‚ÐµÐ»ÑŒ . Ð¸Ð»Ð¸ , Ð¸ Â«Ð»ÑŽÑ„Ñ‚Â» Ð¿Ñ€Ð¾Ð±ÐµÐ»Ð¾Ð²)
          - Ñ‚ÐµÐºÑÑ‚: ÑÐºÐ»ÐµÐ¹ÐºÐ° Ñ‚Ð¾ÐºÐµÐ½Ð¾Ð² Ñ Ð´Ð¾Ð¿ÑƒÑÑ‚Ð¸Ð¼Ñ‹Ð¼Ð¸ Ñ€Ð°Ð·Ð´ÐµÐ»Ð¸Ñ‚ÐµÐ»ÑÐ¼Ð¸
          - mojibake-Ð²Ð°Ñ€Ð¸Ð°Ð½Ñ‚Ñ‹ Ñ‚ÐµÐºÑÑ‚Ð°
        """
        if answer is None:
            return []

        as_str = str(answer).strip()
        as_norm = _norm(as_str)

        # â€” Ð§Ð˜Ð¡Ð›Ð â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”
        # Ñ†ÐµÐ»Ð¾Ðµ (Ñ Ð²Ð¾Ð·Ð¼Ð¾Ð¶Ð½Ñ‹Ð¼Ð¸ Ñ€Ð°Ð·Ð´ÐµÐ»Ð¸Ñ‚ÐµÐ»ÑÐ¼Ð¸ Ñ‚Ñ‹ÑÑÑ‡ Ð¿Ñ€Ð¾Ð±ÐµÐ»Ð°Ð¼Ð¸/NBSP/thin space)
        if re.fullmatch(r"\d+(?:[ \u00A0\u2009]\d+)*", as_norm):
            digits = re.sub(r"[ \u00A0\u2009]+", "", as_norm)
            # Ñ‚Ð¾Ñ‡Ð½Ð¾Ðµ Ð±ÐµÐ· Ð¿Ñ€Ð¸Ð»ÐµÐ³Ð°ÑŽÑ‰Ð¸Ñ… Ñ†Ð¸Ñ„Ñ€
            p1 = rf"(?<!\d){re.escape(digits)}(?!\d)"
            # Â«Ñ€Ð°Ð·Ñ€ÐµÐ¶Ñ‘Ð½Ð½Ñ‹Ð¹Â» Ð²Ð°Ñ€Ð¸Ð°Ð½Ñ‚: 3\\s*0\\s*0
            spaced = r"".join([re.escape(d) + r"[\s\u00A0\u2009]*" for d in digits])
            p2 = rf"(?<!\d){spaced}(?!\d)"
            return [p1, p2]

        # Ð´ÐµÑÑÑ‚Ð¸Ñ‡Ð½Ð¾Ðµ: Ð´Ð¾Ð¿ÑƒÑÐºÐ°ÐµÐ¼ , Ð¸Ð»Ð¸ . Ñ Â«Ð»ÑŽÑ„Ñ‚Ð¾Ð¼Â», Ð¸ Ñ€Ð°Ð·Ð´ÐµÐ»Ð¸Ñ‚ÐµÐ»Ð¸ Ñ‚Ñ‹ÑÑÑ‡ Ð² Ñ†ÐµÐ»Ð¾Ð¹ Ñ‡Ð°ÑÑ‚Ð¸
        if re.fullmatch(r"\d+(?:[ \u00A0\u2009]\d+)*[ ,.]\s*\d+", as_norm):
            canon = as_norm.replace(",", ".")
            head = re.sub(r"[ \u00A0\u2009]+", "", canon).split(".", 1)[0]
            tail = canon.split(".", 1)[1].strip()
            # Ð¿Ð»Ð¾Ñ‚Ð½Ñ‹Ð¹ Ð²Ð°Ñ€Ð¸Ð°Ð½Ñ‚
            p1 = rf"(?<!\d){re.escape(head)}[\s\u00A0\u2009]*[.,][\s\u00A0\u2009]*{re.escape(tail)}(?!\d)"
            # Â«Ñ€Ð°Ð·Ñ€ÐµÐ¶Ñ‘Ð½Ð½Ð°ÑÂ» Ñ†ÐµÐ»Ð°Ñ Ð¸ Ð´Ñ€Ð¾Ð±Ð½Ð°Ñ Ñ‡Ð°ÑÑ‚Ð¸
            spaced_head = "".join([re.escape(d) + r"[\s\u00A0\u2009]*" for d in head])
            spaced_tail = "".join([re.escape(d) + r"[\s\u00A0\u2009]*" for d in tail])
            p2 = rf"(?<!\d){spaced_head}[\s\u00A0\u2009]*[.,][\s\u00A0\u2009]*{spaced_tail}(?!\d)"
            return [p1, p2]

        # â€” Ð¢Ð•ÐšÐ¡Ð¢ â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”
        s = as_norm
        if not s:
            return []
        base_patterns: List[str] = []

        # Ð¾ÑÐ½Ð¾Ð²Ð½Ð¾Ð¹ Ð¸ Â«ÑˆÐ¸Ñ€Ð¾ÐºÐ¸Ð¹Â» Ð·Ð°Ð·Ð¾Ñ€
        base_patterns.append(_tokens_pattern(s, gap=_GAP))
        base_patterns.append(_tokens_pattern(s, gap=r"\W*"))

        # mojibake-Ð²Ð°Ñ€Ð¸Ð°Ð½Ñ‚Ñ‹
        for v in _mojibake_variants(s):
            v_norm = _norm(v)
            base_patterns.append(_tokens_pattern(v_norm, gap=_GAP))
            base_patterns.append(_tokens_pattern(v_norm, gap=r"\W*"))

        # ÑƒÐ´Ð°Ð»Ð¸Ð¼ Ð´ÑƒÐ±Ð»Ð¸ÐºÐ°Ñ‚Ñ‹, ÑÐ¾Ñ…Ñ€Ð°Ð½ÑÑ Ð¿Ð¾Ñ€ÑÐ´Ð¾Ðº
        seen = set()
        uniq = []
        for p in base_patterns:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return uniq

    def any_match(self, text: str, patterns: List[str]) -> Tuple[bool, str]:
        """ÐŸÑ€Ð¾Ð²ÐµÑ€ÑÐµÐ¼, ÑÐ¾Ð²Ð¿Ð°Ð´Ð°ÐµÑ‚ Ð»Ð¸ Ñ‚ÐµÐºÑÑ‚ Ñ Ð»ÑŽÐ±Ñ‹Ð¼ Ð¿Ð°Ñ‚Ñ‚ÐµÑ€Ð½Ð¾Ð¼."""
        t = _norm(text)
        for pat in patterns:
            if re.search(pat, t, flags=re.IGNORECASE | re.UNICODE | re.DOTALL):
                return True, pat
        return False, ""

# ÐŸÑƒÐ±Ð»Ð¸Ñ‡Ð½Ñ‹Ð¹ namespace-Ð¾Ð±ÑŠÐµÐºÑ‚
relaxed_answer_match = RelaxedAnswerMatch()

__all__ = ["relaxed_answer_match", "RelaxedAnswerMatch"]

