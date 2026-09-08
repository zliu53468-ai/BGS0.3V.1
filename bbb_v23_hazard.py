"""JSON-safe public wrapper for BBB V23_SHORT_X_DYNAMIC_HAZARD.

The original V23 calculation core is preserved byte-for-byte in
bbb_v23_hazard_core.py. This wrapper only normalizes diagnostic mapping keys
before the prediction payload is stored by the LINE session JSON store.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import bbb_v23_hazard_core as _core
from bbb_v23_hazard_core import *  # noqa: F401,F403

VERSION = _core.VERSION
GAP_SCALE = _core.GAP_SCALE
_JSON_KEY_TYPES = (str, int, float, bool, type(None))


def _json_safe(value: Any) -> Any:
    """Recursively make diagnostic payloads safe for json.dump.

    V23 Big-Road internals use tuple (row, col) keys inside the occupied lookup
    table. Those keys are valid for Python calculations but invalid as JSON
    object keys. Stringifying only unsupported mapping keys leaves all formal
    prediction values unchanged.
    """
    if isinstance(value, Mapping):
        safe: dict[Any, Any] = {}
        for key, item in value.items():
            safe_key = key if isinstance(key, _JSON_KEY_TYPES) else str(key)
            safe[safe_key] = _json_safe(item)
        return safe
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_json_safe(item) for item in value)
    return value


def predict_v23_bandit(*, history: Any, shoe_context: Mapping[str, Any] | None, scope_key: str) -> dict[str, Any]:
    """Run the untouched V23 core, then sanitize diagnostics for LINE JSON."""
    result = _core.predict_v23_bandit(
        history=history,
        shoe_context=shoe_context,
        scope_key=scope_key,
    )
    return _json_safe(result)


def __getattr__(name: str) -> Any:
    return getattr(_core, name)


__all__ = list(getattr(_core, "__all__", []))
if "predict_v23_bandit" not in __all__:
    __all__.append("predict_v23_bandit")
if "VERSION" not in __all__:
    __all__.append("VERSION")
if "GAP_SCALE" not in __all__:
    __all__.append("GAP_SCALE")
