# DQ Summary Checklist

## Page triage
- [ ] Порог text_min_len подтверждён
- [ ] Порог badchars_max_ratio подтверждён
- [ ] Правила corrupt-font подтверждены

## OCR policy
- [ ] Языки rus+kaz достаточно
- [ ] Профиль A/B и fallback логика понятны
- [ ] cer_max и suspect_char_ratio_max подтверждены
- [ ] Параллелизм соответствует машине

## Tables policy
- [ ] Порядок Lattice→Stream→Scan — ок
- [ ] struct_valid_min, numeric_ratio_min подтверждены
- [ ] Правила risky_table ок

## Normalization
- [ ] Переносы с дефисом склеиваем
- [ ] Тысячные пробелы/десятичные разделители — ок
- [ ] % не теряем/не приписываем
- [ ] Политика дат RU/KZ — ок

## DQ-ворота
- [ ] Матрица «метрика→порог→действие» заполнена
- [ ] Цель public answer findability зафиксирована

## Go/No-Go
- [ ] Все спецификации прочитаны и приняты
- [ ] Конфиги отражают принятые пороги
- [ ] Готовы перейти к реализации ingest/OCR/tables/normalize (Задание №3)

**Принято/Не принято:** __________________
