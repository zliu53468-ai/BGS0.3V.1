import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from config import DEFAULT_GAME, DEFAULT_TABLE, SESSION_EXPIRE_SECONDS

@dataclass
class UserSession:
    user_id: str
    phase: str = "idle"
    game: str = DEFAULT_GAME
    table: str = DEFAULT_TABLE
    active: bool = False
    last_round: Optional[Dict[str, Any]] = None
    rounds: List[Dict[str, Any]] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

class SessionStore:
    def __init__(self):
        self._sessions: Dict[str, UserSession] = {}

    def get(self, user_id: str) -> UserSession:
        now = time.time()
        sess = self._sessions.get(user_id)

        if sess is None:
            sess = UserSession(user_id=user_id)
            self._sessions[user_id] = sess
            return sess

        if now - sess.updated_at > SESSION_EXPIRE_SECONDS:
            sess = UserSession(user_id=user_id, game=sess.game, table=sess.table)
            self._sessions[user_id] = sess
            return sess

        sess.updated_at = now
        return sess

    def reset(self, user_id: str, keep_setting: bool = True) -> UserSession:
        old = self._sessions.get(user_id)
        if keep_setting and old:
            sess = UserSession(user_id=user_id, game=old.game, table=old.table)
        else:
            sess = UserSession(user_id=user_id)
        self._sessions[user_id] = sess
        return sess

    def all_count(self) -> int:
        return len(self._sessions)

store = SessionStore()
