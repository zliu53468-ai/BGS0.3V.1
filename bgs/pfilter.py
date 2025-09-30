"""OutcomePF fallback匯出模組。

在標準情境下，`server.py` 會從 :mod:`bgs.pfilter` 匯入真正的
`OutcomePF` 粒子濾波器；但若部署環境無法把 `bgs` 套件放到
`sys.path`（或是以單檔啟動），就會退回到本地的 `pfilter.py`。

之前的版本只是一層 re-export，遇到上述情況仍然會匯入失敗，最後
觸發 `DummyPF`，導致預測永遠是固定的 48%/47%/5%。為了讓備援生效，
這裡在 re-export 失敗時會直接提供一份純粹的 `OutcomePF`
實作，介面與 `bgs.pfilter.OutcomePF` 完全相容。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

try:  # 優先使用套件版本，功能完整且與 bgs 內部模組共用狀態
    from bgs.pfilter import OutcomePF as _OutcomePF  # type: ignore
except Exception:  # pragma: no cover - 僅在部署環境缺少 bgs 套件時觸發
    import random
    from dataclasses import dataclass
    from typing import List

    import numpy as np

    @dataclass
    class _Particle:
        p: np.ndarray  # shape=(3,), sum=1
        w: float

    def _normalize(vec: np.ndarray) -> np.ndarray:
        vec = np.clip(vec, 1e-12, None)
        total = float(vec.sum())
        if total <= 0:
            return np.array([1 / 3, 1 / 3, 1 / 3], dtype=vec.dtype)
        return vec / total

    class _OutcomePF:
        """精簡版粒子濾波實作，介面與原版完全一致。"""

        def __init__(
            self,
            decks: int = 8,
            seed: int = 42,
            n_particles: int = 200,
            sims_lik: int = 80,
            resample_thr: float = 0.5,
            backend: str = "mc",
            dirichlet_eps: float = 0.003,
            **kwargs,
        ) -> None:
            self.decks = int(decks)
            self.backend = str(backend).lower()
            self.rng = np.random.default_rng(int(seed))
            random.seed(int(seed))

            self.n = max(1, int(n_particles))
            self.n_particles = self.n  # 提供 server.py 所需的屬性
            self.sims_lik = int(sims_lik)
            self.resample_thr = float(resample_thr)
            self.alpha0 = max(1e-6, float(dirichlet_eps))

            self.counts = np.zeros(3, dtype=np.float64)
            self.particles: List[_Particle] = []
            base_alpha = [self.alpha0] * 3
            for _ in range(self.n):
                p = self.rng.dirichlet(base_alpha).astype(np.float64)
                self.particles.append(_Particle(p=p, w=1.0 / self.n))

        def update_outcome(self, outcome: int) -> None:
            if outcome not in (0, 1, 2):
                return
            self.counts[outcome] += 1.0

            weights = np.fromiter(
                (max(1e-12, particle.p[outcome]) for particle in self.particles),
                dtype=np.float64,
                count=self.n,
            )
            total = float(weights.sum())
            if total > 0:
                weights /= total
            else:
                weights.fill(1.0 / self.n)

            for particle, w in zip(self.particles, weights):
                particle.w = float(w)

            ess = 1.0 / float((weights ** 2).sum())
            if ess / self.n < self.resample_thr:
                self._resample()

            alpha_post = self.alpha0 + self.counts
            for particle in self.particles:
                particle.p = self.rng.dirichlet(alpha_post).astype(np.float64)

        def _resample(self) -> None:
            weights = np.array([particle.w for particle in self.particles], dtype=np.float64)
            total = float(weights.sum())
            if total <= 0:
                weights.fill(1.0 / self.n)
            else:
                weights /= total
            idx = self.rng.choice(self.n, size=self.n, replace=True, p=weights)
            self.particles = [
                _Particle(p=self.particles[i].p.copy(), w=1.0 / self.n)
                for i in idx
            ]

        def predict(self, sims_per_particle: int = 0) -> np.ndarray:
            alpha_post = self.alpha0 + self.counts
            base = alpha_post / alpha_post.sum()

            ps = np.stack([particle.p for particle in self.particles], axis=0)
            ws = np.array([particle.w for particle in self.particles], dtype=np.float64)
            ws = ws / max(1e-12, float(ws.sum()))
            mix = (ps * ws[:, None]).sum(axis=0)

            return _normalize(0.6 * mix + 0.4 * base).astype(np.float32)

    OutcomePF = _OutcomePF
else:  # 匯入成功時直接轉出原版類別
    OutcomePF = _OutcomePF

if TYPE_CHECKING:  # 僅供型別檢查工具使用
    from typing import Type as _Type

    _: _Type[_OutcomePF]

__all__ = ["OutcomePF"]
