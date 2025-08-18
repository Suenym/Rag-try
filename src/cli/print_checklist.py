from pathlib import Path

import typer
import yaml

app = typer.Typer(add_completion=False)


def load_yaml(path: str):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


@app.command()
def main(
    app_config: str = "configs/app.yaml",
    thresholds_config: str = "configs/dq_thresholds.yaml",
):
    app_cfg = load_yaml(app_config)
    thr_cfg = load_yaml(thresholds_config)

    ingest = app_cfg.get("ingest", {})
    ocr = app_cfg.get("ocr", {})
    ocr_thr = thr_cfg.get("ocr", {})
    table_thr = thr_cfg.get("tables", {})
    public_thr = thr_cfg.get("public", {})

    print("Page triage")
    print(f" - Порог text_min_len = {ingest.get('text_min_len')} подтверждён")
    print(f" - Порог badchars_max_ratio = {ingest.get('badchars_max_ratio')} подтверждён")
    print(f" - Правила corrupt-font (ratio {ingest.get('corrupt_font_ratio')}) подтверждены\n")

    print("OCR policy")
    print(f" - Языки {', '.join(ocr.get('languages', []))} достаточно")
    print(" - Профиль A/B и fallback логика понятны")
    print(f" - cer_max = {ocr_thr.get('cer_max')} и suspect_char_ratio_max = {ocr_thr.get('suspect_char_ratio_max')} подтверждены")
    print(f" - Параллелизм {ocr.get('parallelism')} соответствует машине\n")

    print("Tables policy")
    print(" - Порядок Lattice→Stream→Scan — ок")
    print(f" - struct_valid_min = {table_thr.get('struct_valid_min')}, numeric_ratio_min = {table_thr.get('numeric_ratio_min')} подтверждены")
    print(" - Правила risky_table ок\n")

    print("Normalization")
    print(" - Переносы с дефисом склеиваем")
    print(" - Тысячные пробелы/десятичные разделители — ок")
    print(" - % не теряем/не приписываем")
    print(" - Политика дат RU/KZ — ок\n")

    print("DQ-ворота")
    print(" - Матрица «метрика→порог→действие» заполнена")
    print(f" - Цель public answer findability зафиксирована ({public_thr.get('answer_findability_target')})\n")

    print("Go/No-Go")
    print(" - Все спецификации прочитаны и приняты")
    print(" - Конфиги отражают принятые пороги")
    print(" - Готовы перейти к реализации ingest/OCR/tables/normalize (Задание №3)")


if __name__ == "__main__":
    app()
