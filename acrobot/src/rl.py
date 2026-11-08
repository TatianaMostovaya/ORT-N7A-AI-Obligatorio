"""Q-learning tabular y Dyna-Q con modelo de transiciones empíricas."""
from collections import defaultdict
import numpy as np

class TabularAgent:
    def __init__(self, n_actions=3, alpha=.1, gamma=.99, planning_steps=0, seed=42):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.n_actions = n_actions
        self.planning_steps = int(planning_steps)
        self.rng = np.random.default_rng(seed)
        self.q = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float64))
        # Cada (s,a) puede producir múltiples s' porque discretizar pierde información.
        self.model = defaultdict(list)
        self.seen_pairs = []

    def act(self, state, epsilon=0.0):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        values = self.q[state]
        best = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best))

    def _update(self, state, action, reward, next_state, terminated):
        # Solo terminated corta el bootstrap: truncated es un límite temporal.
        target = reward if terminated else reward + self.gamma * np.max(self.q[next_state])
        self.q[state][action] += self.alpha * (target - self.q[state][action])

    def learn(self, state, action, reward, next_state, terminated, truncated=False):
        self._update(state, action, reward, next_state, terminated)
        if self.planning_steps == 0:
            return
        pair = (state, action)
        if pair not in self.model:
            self.seen_pairs.append(pair)
        # Modelo acotado: hasta 8 resultados por par (s,a).
        # Se reemplaza una muestra aleatoria cuando se supera el límite.
        outcomes = self.model[pair]
        transition = (float(reward), next_state, bool(terminated))
        if len(outcomes) < 8:
            outcomes.append(transition)
        else:
            outcomes[int(self.rng.integers(8))] = transition
        for _ in range(self.planning_steps):
            sampled_pair = self.seen_pairs[int(self.rng.integers(len(self.seen_pairs)))]
            options = self.model[sampled_pair]
            sample = options[int(self.rng.integers(len(options)))]
            self._update(*sampled_pair, *sample)

    def export_q(self):
        return {state: values.copy() for state, values in self.q.items()}
