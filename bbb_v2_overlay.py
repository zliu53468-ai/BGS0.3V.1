"""Prediction-only adapter: BBB V2 research direction into existing BGS output fields.

The official B/P/T-only V2 BET action is SKIP. The screen may display a
research-direction signal, exactly as BBB's quant256_online_panel.js does,
but this never authorizes a positive stake or mislabels virtual EV as real.
"""
from __future__ import annotations

from typing import Any, Mapping

from bbb_v2_runtime import SOURCE_COMMIT, VERSION, predict_v2


def overlay_v2(
    result: dict[str, Any], *, history: list[str],
    core_policy: Mapping[str, Any], bankroll: float
) -> dict[str, Any]:
    source = predict_v2(history, scope_key=str(core_policy.get("scope_key") or ""))
    research = dict(source["research_signal"])
    probabilities = dict(source["physical_ev"]["simulated_probability"])
    if any(not isinstance(probabilities.get(k), (int, float)) for k in ("B", "P", "T")):
        raise RuntimeError("BBB V2 simulated probabilities unavailable")
    # LINE screenshot supplies B/P/T history, NOT verified per-card ranks.
    if source.get("proxy_only") is not True or source.get("validated_edge") is not False:
        raise RuntimeError("BBB V2 observed-card evidence contract changed")

    candidate = str(source["display_direction"])
    available = research.get("available") is True
    shown = candidate if available else "Skip"
    if candidate not in ("B", "P") or source.get("action") != "SKIP":
        raise RuntimeError("BBB V2 official BET/Skip contract changed")
    label = "觀望" if shown == "Skip" else ("莊" if shown == "B" else "閒")
    # Unlike legacy 5-30% staking, B/P/T research does NOT prove positive EV.
    money = dict(result.get("money_management") or {})
    money.update({
        "direction": "Skip", "bankroll": bankroll,
        "final_bet_ratio": 0.0, "bet_percentage": 0.0, "bet_amount": 0.0,
        "kelly_fraction": 0.0, "bet_allowed": False, "mandatory_bet": False,
        "virtual_ev": float(source["ev_banker"] if candidate == "B" else source["ev_player"]),
        "reason": "bbb_v2_bpt_proxy_only_no_verified_bet",
    })
    components = dict(result.get("component_probabilities") or {})
    components["bbb_v2_physics"] = probabilities
    dynamic = dict(result.get("dynamic_prediction_policy") or {})
    dynamic.update({
        "version": VERSION, "formal_direction_source": "bbb_v2_physical_ev",
        "physics_primary": True, "linucb_primary": False,
        "particle_count": 256, "likelihood_proposals": 384,
        "simulations_per_particle": 160,
        "ocr_or_screen_flow_modified": False
    })
    v2 = {
        "active": True, "source_commit": SOURCE_COMMIT, "version": VERSION,
        "history_length": source["history_length"],
        "config": source["config"],
        "action": source["action"], "official_bet_allowed": False,
        "display_direction": candidate,
        "research_signal": research,
        "gate_phase": source["gate_phase"], "dynamic_gate": source["dynamic_gate"],
        "effective_research_gate": source["effective_research_gate"],
        "win_streak": source["win_streak"], "last_settlement": source["last_settlement"],
        "hit_rate": source["hit_rate"], "signal_rate": source["signal_rate"],
        "ev_banker": source["ev_banker"], "ev_player": source["ev_player"],
        "ev_gap_deviation": source["ev_gap_deviation"],
        "physical_ev": source["physical_ev"],
        "proxy_only": True, "validated_edge": False
    }
    reason = (
        "BBB V2 模擬物理 EV；"
        f"研究方向={label}；偏移={float(source['ev_gap_deviation']):+.5f}；"
        f"有效門檻={float(source['effective_research_gate']):.5f}；"
        "只有 B/P/T，未驗證實際殘牌及下注正期望值。"
    )
    result.update({
        "engine": "BBB_PHYSICS_FORECAST_FIRST_V2",
        "model_version": VERSION, "system_model_version": VERSION,
        "model_variant": "V2_PF256_L384_M160_DYNAMIC_RESEARCH",
        "model_core": "bbb_v2_physics_primary", "primary_model": "BBB_V2_PHYSICS_EV",
        "decision_pipeline": "blind_pf256_l384_to_physical_ev_m160_to_dynamic_research_gate",
        "probabilities": probabilities,
        "banker_rate": round(float(probabilities["B"]) * 100, 2),
        "player_rate": round(float(probabilities["P"]) * 100, 2),
        "tie_rate": round(float(probabilities["T"]) * 100, 2),
        "recommend": shown, "action": shown, "direction": shown,
        "internal_recommend": shown, "internal_action": shown,
        "next_round_direction": shown, "adaptive_only_direction": shown,
        "recommend_text": label, "action_text": label,
        "direction_text": label, "next_round_direction_text": label,
        "formal_action": "Skip", "formal_bet_allowed": False,
        "signal_allowed": available, "risk_gate_open": False,
        "mandatory_bet": False, "bet_allowed": False,
        "skip": not available, "force_observe": not available,
        "skip_reason": "" if available else "bbb_v2_dynamic_research_gate",
        "signal_status_code": "BBB_V2_RESEARCH_SIGNAL" if available else "BBB_V2_RESEARCH_SKIP",
        "signal_status_text": f"下一局研究方向：{label}｜非實際下注資格",
        "signal_reason": reason, "internal_signal_reason": reason,
        "direction_source": "bbb_v2_physical_ev",
        "formal_direction_source": "bbb_v2_physical_ev",
        "linucb_direction_weight": 0.0, "physics_particle_direction_weight": 1.0,
        "road_direction_weight": 0.0, "road_context_direction_weight": 0.0,
        "confidence": 0.0, "raw_model_confidence": 0.0,
        "confidence_label": "模擬，非勝率",
        "confidence_calibration": {
            "applied": False, "raw_confidence": 0.0, "final_confidence": 0.0,
            "direction_override": True,
            "semantics": "unvalidated_bpt_virtual_ev_not_hit_probability"
        },
        "banker_expected_return": float(source["ev_banker"]),
        "player_expected_return": float(source["ev_player"]),
        "banker_ev": float(source["ev_banker"]),
        "player_ev": float(source["ev_player"]),
        "direction_edge": float(source["ev_gap_deviation"]),
        "direction_edge_percent": float(source["ev_gap_deviation"]) * 100,
        "selected_expected_return": 0.0, "selected_expected_return_percent": 0.0,
        "final_probs": probabilities, "direction_probs": probabilities,
        "economic_probs": probabilities,
        "final_probability": float(probabilities.get(candidate, 0.0)),
        "economic_probability_for_direction": 0.0,
        "component_probabilities": components,
        "money_management": money, "kelly_fraction": 0.0,
        "pre_tie_adjusted_ratio": 0.0, "adjusted_ratio": 0.0,
        "final_bet_ratio": 0.0, "bet_percentage": 0.0,
        "suggested_bet_amount": 0.0, "bet_amount": 0.0, "bet_multiplier": 0.0,
        "bet_level_text": "研究用模擬 EV，不提供配注",
        "bet_reason": reason,
        "decision": shown, "decision_text": label,
        "probability_semantics": "simulated_bpt_physical_outcome_distribution_not_validated",
        "decision_gate": {
            "decision": "Skip", "display_direction": shown, "allowed": False,
            "reason": "no_complete_observed_card_evidence",
            "expected_net_ev": float(source["ev_banker"] if candidate == "B" else source["ev_player"]),
            "penalty_observe": not available,
        },
        "fusion_decision": {
            "direction": shown, "probabilities": probabilities,
            "method": "bbb_v2_physical_ev_research",
            "linucb_weight": 0.0, "road_pattern_weight": 0.0,
            "shoe_composition_weight": 1.0,
            "details": {"physical_weight": 5, "road_diagnostic_weight": .1,
                        "road_can_flip_direction": False}
        },
        "dynamic_prediction_policy": dynamic, "dynamic_policy_version": VERSION,
        "bbb_v2": v2,
    })
    return result
