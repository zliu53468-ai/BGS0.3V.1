#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# fix_derived_series_incremental.py
#
# 目的：
# 修正 predictor.py 下三路紅藍生成邏輯。
#
# 原本問題：
# _derived_series(layout, offset) 使用「最後完整大路版面」回算過去所有下三路顏色。
# 這會讓後面新開的格子反向影響前面已經出現過的紅藍。
# 實際百家樂路紙是逐局生成，不會重畫過去的下三路顏色。
#
# 修正方式：
# 1. 在 _build_big_road() 回傳資料中加入 sequence，保留當時 B/P 序列。
# 2. 把 _derived_series() 改成「逐局 prefix 生成」：
#    第 1 局、第 2 局、第 3 局...一路逐步建立大路，只取當局新增的下三路顏色。
# 3. 其他模型不動刀。Road Lifecycle / Memory / Rhythm / Pattern Replay / ML / DeepSeek 都不替換。
#
# 使用方式：
# python fix_derived_series_incremental.py
# python -m py_compile predictor.py

from pathlib import Path
from datetime import datetime
import re
import sys


PREDICTOR_PATH = Path("predictor.py")


NEW_DERIVED_SERIES_BLOCK = r'''
def _derived_series(layout: Dict[str, Any], offset: int) -> List[int]:
    # 下三路紅藍序列必須「逐局生成」，不能用最後完整大路版面回推。
    #
    # 舊版問題：
    #   直接拿最後 layout 裡的 grid 去計算每個歷史 pos。
    #   但後面新開的格子，可能會讓前面某一顆的「有無」判斷被改變。
    #   實戰路紙不會回頭重畫，所以這會造成下三路反應慢或方向失真。
    #
    # 新版：
    #   如果 layout 內有 sequence，就用每一口 prefix 逐局建立大路，
    #   只取當下新增那一顆的紅/藍，確保不被未來格子污染。
    seq = layout.get("sequence")
    if seq:
        clean_seq = [x for x in seq if x in {"B", "P"}]
        cache_key = (int(offset), "".join(clean_seq))

        cache = getattr(_derived_series, "_cache", None)
        if cache is None:
            cache = {}
            setattr(_derived_series, "_cache", cache)

        if cache_key in cache:
            return list(cache[cache_key])

        series: List[int] = []
        for i in range(1, len(clean_seq) + 1):
            prefix = clean_seq[:i]
            partial_layout = _build_big_road(prefix)
            positions = partial_layout.get("positions", []) or []
            if not positions:
                continue
            pos = positions[-1]
            color = _derived_color_at(partial_layout, pos, offset)
            if color != 0:
                series.append(color)

        if len(cache) > 500:
            cache.clear()
        cache[cache_key] = list(series)
        return series

    # 相容 fallback：若舊資料沒有 sequence，才使用舊版 final-layout 算法。
    # 正常 predictor.py 由 _build_big_road() 產生的 layout 都會有 sequence。
    series = []
    for pos in layout.get("positions", []):
        color = _derived_color_at(layout, pos, offset)
        if color != 0:
            series.append(color)
    return series
'''.strip()


def fail(message: str) -> None:
    print(f"[ERROR] {message}")
    sys.exit(1)


def replace_between(text: str, start_marker: str, end_marker: str, new_block: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        fail(f"找不到起點：{start_marker}")

    end = text.find(end_marker, start)
    if end < 0:
        fail(f"找不到終點：{end_marker}")

    return text[:start] + new_block.rstrip() + "\n\n\n" + text[end:]


def add_sequence_to_build_big_road(text: str) -> str:
    # 只在 _build_big_road 的 return dict 裡新增 sequence。
    if '"sequence": [x for x in non_tie if x in {"B", "P"}],' in text:
        return text

    target = '''return {
        "rows": rows,
        "grid": grid,'''

    if target not in text:
        fail("找不到 _build_big_road() 的 return 區塊，無法加入 sequence。")

    replacement = '''return {
        "rows": rows,
        "sequence": [x for x in non_tie if x in {"B", "P"}],
        "grid": grid,'''

    return text.replace(target, replacement, 1)


def main() -> None:
    if not PREDICTOR_PATH.exists():
        fail("目前資料夾找不到 predictor.py，請把本腳本放在 predictor.py 同一層。")

    text = PREDICTOR_PATH.read_text(encoding="utf-8")

    backup_path = PREDICTOR_PATH.with_name(f"predictor.backup-derived-incremental-{datetime.now():%Y%m%d-%H%M%S}.py")
    backup_path.write_text(text, encoding="utf-8")

    text = add_sequence_to_build_big_road(text)
    text = replace_between(
        text,
        "def _derived_series(",
        "def _color_stats(",
        NEW_DERIVED_SERIES_BLOCK,
    )

    PREDICTOR_PATH.write_text(text, encoding="utf-8")

    print("[OK] predictor.py 已修正下三路為逐局生成算法。")
    print(f"[OK] 原檔備份：{backup_path.name}")
    print("[OK] 本腳本只修改 _build_big_road() 回傳 sequence 與 _derived_series()。")
    print("[OK] 其他模型沒有動刀。")
    print("[OK] 請執行：python -m py_compile predictor.py")


if __name__ == "__main__":
    main()
