import re
from typing import Optional, Tuple

def parse_points(text: str) -> Optional[Tuple[int, int]]:
    """
    支援：
    65
    6 5
    6,5
    6/5
    開始分析65

    回傳：
    (player_point, banker_point)
    """
    t = text.strip()
    t = t.replace("開始分析", "").strip()

    m = re.fullmatch(r"([0-9])\s*[,，/／\-\s]\s*([0-9])", t)
    if m:
        return int(m.group(1)), int(m.group(2))

    m = re.fullmatch(r"([0-9])([0-9])", t)
    if m:
        return int(m.group(1)), int(m.group(2))

    return None

def looks_like_table_id(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]{1,5}[0-9]{1,4}", text.strip()))
