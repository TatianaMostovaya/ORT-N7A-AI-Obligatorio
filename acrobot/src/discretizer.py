"""Discretización de observaciones Acrobot-v1."""
import numpy as np

# límites oficiales de la observación de Gymnasium Acrobot-v1
SIX_LOW = np.array([-1, -1, -1, -1, -4 * np.pi, -9 * np.pi], dtype=float)
SIX_HIGH = -SIX_LOW
ANGLE_LOW = np.array([-np.pi, -np.pi, -4 * np.pi, -9 * np.pi], dtype=float)
ANGLE_HIGH = -ANGLE_LOW

class Discretizer:
    def __init__(self, mode="angles", bins=8, edges=None):
        if mode not in ("six", "angles", "nonuniform"):
            raise ValueError("mode debe ser six, angles o nonuniform")
        if bins < 2:
            raise ValueError("bins >= 2")
        self.mode = mode
        self.bins = int(bins)
        self.low, self.high = (SIX_LOW, SIX_HIGH) if mode == "six" else (ANGLE_LOW, ANGLE_HIGH)
        if edges is None:
            if mode == "nonuniform":
                # Más cortes cerca de cero en las velocidades.
                t = np.linspace(-1, 1, bins + 1)[1:-1]
                self.edges = [np.linspace(lo, hi, bins + 1)[1:-1] if i < 2
                              else (hi * np.sign(t) * np.abs(t) ** 2)
                              for i, (lo, hi) in enumerate(zip(self.low, self.high))]
            else:
                self.edges = [np.linspace(lo, hi, bins + 1)[1:-1]
                              for lo, hi in zip(self.low, self.high)]
        else:
            self.edges = [np.asarray(e, dtype=float) for e in edges]
        self.n_states_theoretical = self.bins ** len(self.edges)

    def transform(self, obs):
        obs = np.asarray(obs, dtype=float)
        if obs.shape != (6,):
            raise ValueError(f"Observación inválida: {obs.shape}")
        if self.mode == "six":
            values = obs
        else:
            # cos1,sin1,cos2,sin2,vel1,vel2
            values = np.array([np.arctan2(obs[1], obs[0]),
                               np.arctan2(obs[3], obs[2]), obs[4], obs[5]])
        return tuple(int(np.searchsorted(edge, value, side="right"))
                     for edge, value in zip(self.edges, values))
