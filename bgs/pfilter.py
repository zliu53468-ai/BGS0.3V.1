# -*- coding: utf-8 -*-
"""pfilter.py — Outcome Particle Filter (independent mode)"""
import os
import numpy as np

# Configuration from environment
HISTORY_MODE = os.getenv("HISTORY_MODE", "0").lower() not in ("0", "false", "no")
# We assume SKIP_TIE_UPD and other parameters might be needed if history mode on
SKIP_TIE_UPD = os.getenv("SKIP_TIE_UPD", "1").lower() not in ("0", "false", "no")
SOFT_TAU = float(os.getenv("SOFT_TAU", "2.0"))
TIE_MIN = float(os.getenv("TIE_MIN", "0.05"))
TIE_MAX = float(os.getenv("TIE_MAX", "0.15"))

class OutcomePF:
    def __init__(self, decks: int = 8, seed: int = 42, n_particles: int = 50, 
                 sims_lik: int = 30, resample_thr: float = 0.5, 
                 backend: str = "mc", dirichlet_eps: float = 0.05):
        # Initialize random seed for reproducibility
        np.random.seed(seed)
        self.decks = decks
        self.n_particles = n_particles
        self.sims_lik = sims_lik
        self._backend = backend
        # If history mode (not default), initialize outcome counts with a small prior
        if HISTORY_MODE:
            # Use theoretical probabilities as a prior distribution
            base_probs = np.array([0.4586, 0.4462, 0.0952], dtype=np.float64)
            # Scale prior counts so initial probabilities align with theoretical
            prior_weight = 1.0
            self._counts = prior_weight * base_probs  # fractional prior counts
            self._total = prior_weight
        else:
            self._counts = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            self._total = 0.0

    @property
    def backend(self) -> str:
        return f"{self._backend}-local"

    def update_outcome(self, outcome: int):
        """Update the filter with the actual outcome (0=莊, 1=閒, 2=和). 
        In independent mode, this does nothing unless history mode is enabled."""
        if not HISTORY_MODE:
            # Independent mode: ignore outcome updates
            return
        if outcome == 2 and SKIP_TIE_UPD:
            # Skip tie updates if configured
            return
        if outcome in (0, 1, 2):
            # Update outcome counts and total
            self._counts[outcome] += 1.0
            self._total += 1.0
            # Log update for debugging
            try:
                import logging
                logging.getLogger("bgs-server").info("OutcomePF 更新: outcome=%s, total_games=%s", outcome, int(self._total))
            except Exception:
                pass

    def predict(self, sims_per_particle: int = 5) -> np.ndarray:
        """Predict outcome probabilities for the next round."""
        if HISTORY_MODE and self._total > 0:
            # If history mode, use the learned distribution (with smoothing) for prediction
            probs = self._counts / self._total
        else:
            # If no history or no outcomes observed, use theoretical base probabilities
            probs = np.array([0.4586, 0.4462, 0.0952], dtype=np.float64)
        # Monte Carlo sampling to introduce variability (simulate draws)
        total_simulations = self.n_particles * sims_per_particle
        counts = np.random.multinomial(total_simulations, probs)
        pred = counts.astype(np.float64) / total_simulations
        # Apply output smoothing (soft_tau)
        pred = pred ** (1.0 / SOFT_TAU)
        pred = pred / pred.sum()
        # Apply tie probability compression
        pT = pred[2]
        if pT < TIE_MIN:
            pred[2] = TIE_MIN
            scale = (1.0 - TIE_MIN) / (pred[0] + pred[1]) if (pred[0] + pred[1]) > 0 else 1.0
            pred[0] *= scale
            pred[1] *= scale
        elif pT > TIE_MAX:
            pred[2] = TIE_MAX
            scale = (1.0 - TIE_MAX) / (pred[0] + pred[1]) if (pred[0] + pred[1]) > 0 else 1.0
            pred[0] *= scale
            pred[1] *= scale
        # Log prediction for debugging
        try:
            import logging
            logging.getLogger("bgs-server").info("OutcomePF 預測: %s (基於 %s 場歷史)", pred, int(self._total))
        except Exception:
            pass
        return pred.astype(np.float32)
