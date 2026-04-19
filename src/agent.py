"""Agente de Q-learning tabular con persistencia de la Q-tabla."""

from __future__ import annotations

import pickle
import random
from collections import defaultdict
from pathlib import Path


class QLearningAgent:
    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.q = defaultdict(lambda: [0.0] * self.n_actions)

    def select_action(self, state) -> int:
        """Política ε-greedy."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return self._argmax(self.q[state])

    def greedy_action(self, state) -> int:
        """Acción greedy (explotación pura)."""
        return self._argmax(self.q[state])

    def update(self, state, action, reward, next_state, done: bool):
        best_next = 0.0 if done else max(self.q[next_state])
        target = reward + self.gamma * best_next
        self.q[state][action] += self.alpha * (target - self.q[state][action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------- Persistencia -------

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "q": dict(self.q),
            "n_actions": self.n_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> "QLearningAgent":
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(
            n_actions=data["n_actions"],
            alpha=data["alpha"],
            gamma=data["gamma"],
            epsilon=data["epsilon"],
            epsilon_min=data["epsilon_min"],
            epsilon_decay=data["epsilon_decay"],
        )
        agent.q = defaultdict(lambda: [0.0] * agent.n_actions, data["q"])
        return agent

    # ------- Utilidades -------

    @staticmethod
    def _argmax(values: list[float]) -> int:
        best = 0
        best_val = values[0]
        for i in range(1, len(values)):
            if values[i] > best_val:
                best_val = values[i]
                best = i
        return best
