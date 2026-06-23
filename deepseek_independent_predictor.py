# deepseek_independent_predictor.py
import os
import json
from typing import Dict, Any, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
USE_DEEPSEEK_INDEPENDENT = os.getenv("USE_DEEPSEEK_INDEPENDENT", "0") == "1"
DEEPSEEK_INDEPENDENT_WEIGHT = float(os.getenv("DEEPSEEK_INDEPENDENT_WEIGHT", "0.5"))


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
    """建立給 DeepSeek 的完整情報報告"""
    gap = abs(player_point - banker_point)
    if gap == 0:
        gap_family = "和點"
    elif gap <= 2:
        gap_family = "極小差距(1-2)"
    elif gap <= 4:
        gap_family = "中小差距(3-4)"
    elif gap <= 7:
        gap_family = "中高差距(5-7)"
    else:
        gap_family = "極大差距(8-9)"

    last_result = "閒" if player_point > banker_point else "莊" if banker_point > player_point else "和"

    report = f"""
=== 當前局情報 ===
閒家點數: {player_point}，莊家點數: {banker_point}，上一局結果: {last_result}
點數差距區間: {gap_family}

=== 點數統計層 (point_db) ===
可用: {point.get('available')}，樣本數: {point.get('sample_size', 0)}
莊家勝率: {point.get('banker_prob', 0.5):.4f}，閒家勝率: {point.get('player_prob', 0.5):.4f}

=== 條件資料庫 (combo_db，點數+補牌情境) ===
可用: {combo.get('available')}，總樣本: {combo.get('sample_size', 0)}
莊家勝率: {combo.get('banker_prob', 0.5):.4f}，閒家勝率: {combo.get('player_prob', 0.5):.4f}
主要情境: {combo.get('top_scenario', '未知')}
匹配記錄數: {len(combo.get('matched_records', []))}

=== 補牌情境 Monte Carlo ===
可用: {comp.get('available')}，模擬樣本: {comp.get('sample_size', 0)}
最可能情境: {comp.get('top_scenario')} (機率 {comp.get('top_scenario_probability', 0):.3f})
次可能情境機率: {comp.get('second_scenario_probability', 0):.3f}
情境熵: {comp.get('scenario_entropy', 1):.3f} (越高代表越不確定)
模擬後莊家勝率: {comp.get('banker_prob', 0.5):.4f}

=== 牌路資料庫 (road_profile，相似路段分佈) ===
可用: {road.get('available')}，樣本數: {road.get('sample_size', 0)}
最相似路段: {road.get('top_road_profile_zh', '無')}
莊家勝率: {road.get('banker_prob', 0.5):.4f}

=== 短牌路模型 (最近4-8口) ===
方向: {micro.get('micro_direction')}，信心: {micro.get('micro_confidence', 0):.2f}
模式: {micro.get('micro_patterns', [])}
龍尾風險: {micro.get('dragon_tail_risk', 0):.2f}

=== AI 模式識別 (近8局結果規律) ===
模式類型: {ai.get('pattern_type')}，強度: {ai.get('pattern_strength', 0):.2f}
模式建議方向: {ai.get('pattern_suggest')}
長龍資訊: {ai.get('streak_side')} {ai.get('streak_count', 0)}口
近期結果序列: {ai.get('history_results', [])}

=== 近期點數趨勢 (最近8局原始點數) ===
{rounds_summary}

請根據以上所有情報，推理下一手莊家勝率。只輸出一個0到1之間的數字，例如0.5234。
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
    """獨立預測：讓 DeepSeek 閱讀完整情報後給出機率"""
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
            max_tokens=10,
        )
        text = response.choices[0].message.content.strip()
        val = float(text)
        return max(0.35, min(0.65, val))
    except Exception:
        return None
