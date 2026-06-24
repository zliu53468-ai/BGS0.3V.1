import os
import json
import re
from typing import Dict, Any, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
USE_DEEPSEEK_INDEPENDENT = os.getenv("USE_DEEPSEEK_INDEPENDENT", "0") == "1"
DEEPSEEK_INDEPENDENT_WEIGHT = float(os.getenv("DEEPSEEK_INDEPENDENT_WEIGHT", "0.78"))


def build_intelligence_report(
    player_point: int,
    banker_point: int,
    point: Dict[str, Any],
    combo: Dict[str, Any],
    road: Dict[str, Any],
    comp: Dict[str, Any],
    micro: Dict[str, Any],
    ai: Dict[str, Any],
    rounds_summary: str,
) -> str:
    gap = abs(player_point - banker_point)
    # 強化點差建議
    if gap == 0:
        gap_family = "和點"
        gap_advice = "點差為0，模型極不穩定。請完全忽略點數統計層，以近期規律（單跳/長龍）和短牌路方向為唯一依據。"
    elif gap <= 2:
        gap_family = f"極小差距({gap})"
        gap_advice = "點差很小，點數層無效。請以近期結果規律和 AI 模式識別為主要依據，點數層最多 20% 權重。"
    elif gap <= 4:
        gap_family = f"中小差距({gap})"
        gap_advice = "有一定方向性，但補牌情境影響顯著。若補牌預測與強勢方矛盾，應降低強勢方信心。"
    elif gap <= 7:
        gap_family = f"中高差距({gap})"
        gap_advice = "強勢方明顯，通常跟隨強勢方。但若短牌路出現轉折信號或補牌預測相反，請勿過度追強。"
    else:
        gap_family = f"極大差距({gap})"
        gap_advice = "⚠️ 極度懸殊，歷史上翻盤率約 8~12%。若近期規律不支持強勢方，或短牌路出現轉折，請優先考慮反向。不要盲目追強。"

    # 補牌影響指南
    draw_guide = """
【補牌情境對勝率的強制調整規則】
- 若下一局 NONE_DRAW 機率最高：勝率直接由點差決定，點差越大越可信。
- 若下一局 PLAYER_DRAW 機率最高：莊家勝率微幅上升（+0.02~0.05），因為閒補牌後點數可能變小。
- 若下一局 BANKER_DRAW 機率最高：閒家勝率微幅上升（+0.02~0.05）。
- 若下一局 BOTH_DRAW 機率最高：不確定性極高，勝率趨近 0.5，請大幅降低信心，以近期規律為主。
請嚴格根據預測的補牌情境機率，動態調整你給出的莊家勝率。
"""

    last_result = "閒" if player_point > banker_point else "莊" if banker_point > player_point else "和"

    # 近期規律
    pattern_type = ai.get("pattern_type", "none")
    pattern_strength = ai.get("pattern_strength", 0)
    pattern_suggest = ai.get("pattern_suggest", "NEUTRAL")
    streak_side = ai.get("streak_side", "NEUTRAL")
    streak_count = ai.get("streak_count", 0)
    pattern_detail = ai.get("pattern_detail", "")
    if streak_count >= 5:
        pattern_detail += " (⚠️長龍尾端，反轉風險極高，請勿追龍)"

    # 各層方向共識
    def side(p):
        return "莊" if p >= 0.5 else "閒"
    layers_sides = {
        "點數層": side(point.get("banker_prob", 0.5)),
        "條件資料庫": side(combo.get("banker_prob", 0.5)),
        "補牌MC": side(comp.get("banker_prob", 0.5)),
        "牌路相似層": side(road.get("banker_prob", 0.5)),
        "短牌路": micro.get("micro_direction", "N"),
        "AI趨勢": ai.get("ai_direction", "N")
    }
    unique_sides = len(set(layers_sides.values()))
    if unique_sides <= 2:
        consensus = "✅ 高度一致"
    elif unique_sides == 3:
        consensus = "⚠️ 有明顯分歧，請以近期規律和點差特性為準"
    else:
        consensus = "🚨 各層方向混亂，必須以近期規律為唯一依據"

    # 下一局補牌情境預測
    next_scenario_text = "無預測資料"
    if os.getenv("USE_NEXT_SCENARIO_PREDICT", "0") == "1":
        try:
            from next_scenario_db import get_next_scenario_probs
            next_probs = get_next_scenario_probs(player_point, banker_point)
            if next_probs:
                items = [f"{k}: {v*100:.1f}%" for k, v in sorted(next_probs.items(), key=lambda x: x[1], reverse=True)]
                next_scenario_text = ", ".join(items)
        except Exception:
            pass

    # 下一局點數分佈（取前8）
    point_dist_text = "無預測資料"
    if "next_point_distribution" in comp and comp["next_point_distribution"]:
        dist = comp["next_point_distribution"]
        items = []
        for d in dist[:8]:
            pp, bp, prob = d["player_point"], d["banker_point"], d["probability"]
            result = "閒勝" if pp > bp else ("莊勝" if bp > pp else "和")
            items.append(f"閒{pp}莊{bp} ({result}) {prob*100:.1f}%")
        point_dist_text = ", ".join(items)

    report = f"""
你是一位專精百家樂數據的 AI 分析師。請根據以下完整的結構化情報，推理下一手莊家勝率。

【補牌影響指南】你必須嚴格遵守以下規則來調整機率：
{draw_guide}

【核心決策框架】按此順序思考：
1. 點差特性：{gap_family}。{gap_advice}
2. 各層共識：{consensus}
   各層方向：{json.dumps(layers_sides, ensure_ascii=False)}
3. 近期規律：模式 {pattern_type} (強度 {pattern_strength:.2f})，建議方向 {pattern_suggest}，{pattern_detail}
   近期長龍：{streak_side} {streak_count}口
4. 下一局補牌預測：{next_scenario_text}
   請根據上述補牌影響指南，計算加權後的莊家勝率。
5. 下一局最可能開出的點數與勝方：{point_dist_text}
6. 參考統計數據：
   - 點數歷史勝率：莊 {point.get('banker_prob',0.5):.4f}（樣本 {point.get('sample_size',0)}）
   - 條件資料庫勝率：莊 {combo.get('banker_prob',0.5):.4f}（樣本 {combo.get('sample_size',0)}）
   - 補牌MC勝率：莊 {comp.get('banker_prob',0.5):.4f}（熵 {comp.get('scenario_entropy',1):.2f}）
   - 短牌路方向：{micro.get('micro_direction')}（信心 {micro.get('micro_confidence',0):.2f}，龍尾風險 {micro.get('dragon_tail_risk',0):.2f}）
7. 近期點數序列：{rounds_summary}

請綜合以上所有資訊，特別是補牌情境對勝率的調整規則，輸出一個 JSON 物件，格式如下：
{{"banker_prob": 0.5234, "reasoning": "簡短說明你為何給出這個機率（20字內）"}}
只輸出 JSON，不要有其他文字。
"""
    return report


def deepseek_independent_predict(
    player_point: int,
    banker_point: int,
    point: Dict[str, Any],
    combo: Dict[str, Any],
    road: Dict[str, Any],
    comp: Dict[str, Any],
    micro: Dict[str, Any],
    ai: Dict[str, Any],
    rounds_summary: str,
) -> Optional[float]:
    if not USE_DEEPSEEK_INDEPENDENT or not DEEPSEEK_API_KEY or OpenAI is None:
        return None

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    prompt = build_intelligence_report(
        player_point, banker_point, point, combo, road, comp, micro, ai, rounds_summary
    )

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        val = float(data.get("banker_prob", 0.5))
        return max(0.35, min(0.65, val))
    except json.JSONDecodeError:
        content = response.choices[0].message.content.strip()
        match = re.search(r"(\d+\.\d+)", content)
        if match:
            val = float(match.group(1))
            return max(0.35, min(0.65, val))
        return None
    except Exception:
        return None
