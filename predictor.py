"""Point-conditioned baccarat predictor using a card-depletion particle filter.

Public compatibility:
    predict(history_or_observations, venue='', room='', shoe_id='', user_id='')

Preferred observation format:
    [{'player': 6, 'banker': 5}, {'player': 2, 'banker': 8}]

Strings such as 'P6B5' or '閒6莊5' are also accepted as individual observations.
"""
from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from particle_filter_points import PointParticleFilter

MAX_FILTERS = max(1, int(os.getenv('PF_MAX_FILTERS', '32')))
RECOMMEND_TIE = os.getenv('PF_RECOMMEND_TIE', '0') == '1'
MAX_DISPLAY_CONFIDENCE = max(0.50, min(0.80, float(os.getenv('PF_MAX_DISPLAY_CONFIDENCE', '0.64'))))

_FILTERS: 'OrderedDict[str, PointParticleFilter]' = OrderedDict()
_FILTER_LOCK = threading.RLock()
_OBS_COUNTS: Dict[str, int] = {}


def _key(user_id: str, venue: str, room: str, shoe_id: str) -> str:
    return '|'.join([user_id or 'anonymous', venue or 'global', room or 'global', shoe_id or 'global'])


def _get_filter(key: str) -> PointParticleFilter:
    with _FILTER_LOCK:
        if key in _FILTERS:
            _FILTERS.move_to_end(key)
            return _FILTERS[key]
        while len(_FILTERS) >= MAX_FILTERS:
            old_key, _ = _FILTERS.popitem(last=False)
            _OBS_COUNTS.pop(old_key, None)
        pf = PointParticleFilter(key)
        _FILTERS[key] = pf
        _OBS_COUNTS[key] = 0
        return pf


def parse_point_observation(value: Any) -> Optional[Dict[str, int]]:
    if isinstance(value, Mapping):
        p = value.get('player', value.get('P', value.get('閒')))
        b = value.get('banker', value.get('B', value.get('莊')))
        try:
            return {'player': int(p) % 10, 'banker': int(b) % 10}
        except Exception:
            return None
    text = str(value or '').strip().upper()
    patterns = [
        r'(?:P|PLAYER|閒|闲)\s*([0-9])\D+(?:B|BANKER|莊|庄)\s*([0-9])',
        r'(?:B|BANKER|莊|庄)\s*([0-9])\D+(?:P|PLAYER|閒|闲)\s*([0-9])',
        r'^\s*([0-9])\s*[,/\- ]\s*([0-9])\s*$',
    ]
    m = re.search(patterns[0], text)
    if m:
        return {'player': int(m.group(1)), 'banker': int(m.group(2))}
    m = re.search(patterns[1], text)
    if m:
        return {'player': int(m.group(2)), 'banker': int(m.group(1))}
    m = re.search(patterns[2], text)
    if m:
        return {'player': int(m.group(1)), 'banker': int(m.group(2))}
    return None


def _clean_observations(values: Union[str, Iterable[Any], None]) -> List[Dict[str, int]]:
    if values is None:
        return []
    if isinstance(values, str):
        chunks = [x for x in re.split(r'[;|\n]+', values) if x.strip()]
    else:
        chunks = list(values)
    result = []
    for item in chunks:
        parsed = parse_point_observation(item)
        if parsed is not None:
            result.append(parsed)
    return result


def _cap_probs(probs: Dict[str, float]) -> Dict[str, float]:
    tie = max(0.0, min(0.30, float(probs.get('T', 0.0))))
    b = max(0.0, float(probs.get('B', 0.0)))
    p = max(0.0, float(probs.get('P', 0.0)))
    non_tie = max(1e-12, b + p)
    b_share, p_share = b / non_tie, p / non_tie
    if max(b_share, p_share) > MAX_DISPLAY_CONFIDENCE:
        if b_share >= p_share:
            b_share, p_share = MAX_DISPLAY_CONFIDENCE, 1.0 - MAX_DISPLAY_CONFIDENCE
        else:
            p_share, b_share = MAX_DISPLAY_CONFIDENCE, 1.0 - MAX_DISPLAY_CONFIDENCE
    return {'B': b_share * (1.0 - tie), 'P': p_share * (1.0 - tie), 'T': tie}


def predict(
    history: Union[str, Iterable[Any]],
    venue: str = '',
    room: str = '',
    shoe_id: str = '',
    user_id: str = '',
) -> Dict[str, Any]:
    observations = _clean_observations(history)
    key = _key(user_id, venue, room, shoe_id)
    pf = _get_filter(key)
    applied = _OBS_COUNTS.get(key, 0)
    if len(observations) < applied:
        reset_uid_model(user_id, venue, room, shoe_id)
        pf = _get_filter(key)
        applied = 0
    updates = []
    for obs in observations[applied:]:
        updates.append(pf.update(obs['player'], obs['banker']))
    _OBS_COUNTS[key] = len(observations)

    raw = pf.predict()
    probs = _cap_probs(raw['probabilities'])
    allowed = ('B', 'P', 'T') if RECOMMEND_TIE else ('B', 'P')
    recommend = max(allowed, key=lambda name: probs[name])
    text = {'B': '莊', 'P': '閒', 'T': '和'}[recommend]
    bp_total = max(1e-12, probs['B'] + probs['P'])
    confidence = max(probs['B'], probs['P']) / bp_total
    edge = abs(probs['B'] - probs['P']) / bp_total
    signal = 'HIGH' if confidence >= 0.58 else 'MEDIUM' if confidence >= 0.54 else 'LOW'
    reason = (
        f"CARD_PF；觀測={len(observations)}；粒子ESS={raw['effective_sample_size']:.1f}；"
        f"下一手模擬={raw['simulations']}；僅以最終點數約束可能牌靴"
    )
    return {
        'ok': True,
        'engine': 'CARD_DEPLETION_POINT_PARTICLE_FILTER',
        'user_id': user_id,
        'venue': venue,
        'room': room,
        'shoe_id': shoe_id,
        'round_no': len(observations) + 1,
        'history_len': len(observations),
        'banker_rate': round(probs['B'] * 100, 1),
        'player_rate': round(probs['P'] * 100, 1),
        'tie_rate': round(probs['T'] * 100, 1),
        'probabilities': probs,
        'recommend': recommend,
        'recommend_text': text,
        'is_observe': False,
        'observe_reason': '',
        'confidence': round(confidence, 4),
        'confidence_pct': round(confidence * 100, 1),
        'decision_edge': round(edge, 6),
        'signal_level': signal,
        'reason': reason,
        'point_particle_filter': raw,
        'applied_updates': updates,
        'ai_used': False,
        'ai_status': 'local_particle_filter',
        'ml_trained': len(observations) > 0,
        'ml_samples': len(observations),
        'tf_available': False,
        'lstm_status': 'disabled_point_pf',
        'global_lstm_status': 'disabled_point_pf',
        'configured_weights': {'particle_filter': 1.0},
        'effective_weights': {'particle_filter': 1.0},
        'debug': None,
    }


def fit_history(history: Union[str, Iterable[Any]], venue: str = '', room: str = '', shoe_id: str = '', user_id: str = '', force: bool = True) -> Dict[str, Any]:
    result = predict(history, venue, room, shoe_id, user_id)
    return {'ok': True, 'history_len': result['history_len'], 'model': 'CARD_DEPLETION_POINT_PARTICLE_FILTER'}


def reset_uid_model(user_id: str, venue: str = '', room: str = '', shoe_id: str = '') -> Dict[str, Any]:
    key = _key(user_id, venue, room, shoe_id)
    with _FILTER_LOCK:
        removed = _FILTERS.pop(key, None) is not None
        _OBS_COUNTS.pop(key, None)
    return {'ok': True, 'removed': int(removed), 'training_key': key}


def clear_model_cache(user_id: Optional[str] = None) -> Dict[str, Any]:
    with _FILTER_LOCK:
        if not user_id:
            n = len(_FILTERS)
            _FILTERS.clear(); _OBS_COUNTS.clear()
            return {'ok': True, 'removed': n}
        prefix = f'{user_id}|'
        keys = [k for k in _FILTERS if k.startswith(prefix)]
        for k in keys:
            _FILTERS.pop(k, None); _OBS_COUNTS.pop(k, None)
        return {'ok': True, 'removed': len(keys)}


def get_model_cache_info() -> Dict[str, Any]:
    return {'size': len(_FILTERS), 'keys': list(_FILTERS.keys()), 'engine': 'CARD_DEPLETION_POINT_PARTICLE_FILTER'}
