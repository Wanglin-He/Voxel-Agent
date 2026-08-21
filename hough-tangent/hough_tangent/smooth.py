"""One Euro filter: little jitter when still, little lag when moving."""

from __future__ import annotations

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * np.pi * max(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class OneEuro:
    """Vectorised One Euro filter over a fixed-shape array."""

    def __init__(
        self, freq: float, min_cutoff: float = 0.8, beta: float = 0.02, d_cutoff: float = 1.0
    ) -> None:
        self.dt = 1.0 / max(freq, 1e-6)
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None

    def reset(self) -> None:
        self._x = None
        self._dx = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._x is None or self._x.shape != x.shape:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            return x.copy()
        dx = (x - self._x) / self.dt
        a_d = _alpha(self.d_cutoff, self.dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        a = 1.0 / (1.0 + (1.0 / (2.0 * np.pi * np.maximum(cutoff, 1e-6))) / self.dt)
        self._x = a * x + (1.0 - a) * self._x
        return self._x.copy()
