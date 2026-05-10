"""Carga la Q-tabla entrenada y prueba varias estrategias de evaluación.

Si results/minigrid/qtable.pkl no existe, entrena primero (mismos
hiperparámetros que verify_minigrid.py).

Pruebas:
  1. Greedy puro con varias semillas para el desempate aleatorio.
  2. Cuasi-greedy con ε ∈ {0.01, 0.05, 0.10}.

Reporta éxito / recompensa / pasos promedio para cada estrategia.
"""

from __future__ import annotations

import random
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import QLearningAgent  # noqa: E402
from src.minigrid.env import DoorKeyEnv  # noqa: E402

QTABLE_PATH = ROOT / "results" / "minigrid" / "qtable.pkl"
N_EPISODES = 10000
MAX_STEPS = 576
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995
ENV_SEED = 0
RANDOM_SEED = 42


def train(env: DoorKeyEnv) -> QLearningAgent:
    random.seed(RANDOM_SEED)
    agent = QLearningAgent(
        n_actions=env.n_actions,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
    )
    rewards = deque(maxlen=500)
    t0 = time.time()
    for ep in range(1, N_EPISODES + 1):
        state, _ = env.reset()
        ep_reward = 0.0
        for _ in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.update(state, action, reward, next_state, done=terminated)
            state = next_state
            ep_reward += reward
            if terminated or truncated:
                break
        agent.decay_epsilon()
        rewards.append(ep_reward)
        if ep in (2000, 4000, 7500, 10000):
            avg = sum(rewards) / len(rewards)
            print(
                f"  ep={ep:>5}  eps={agent.epsilon:.3f}  "
                f"avg_last500={avg:+.3f}  ({time.time() - t0:.0f}s)"
            )
    return agent


def eval_strategy(
    env: DoorKeyEnv,
    agent: QLearningAgent,
    *,
    eps: float,
    rng_seed: int,
    n_eval: int = 100,
) -> tuple[int, float, float]:
    random.seed(rng_seed)
    success = 0
    rewards = []
    steps_list = []
    for _ in range(n_eval):
        state, _ = env.reset()
        ep_reward = 0.0
        terminated = False
        for step in range(MAX_STEPS):
            if eps > 0 and random.random() < eps:
                action = random.randrange(agent.n_actions)
            else:
                action = agent._argmax(agent.q[state])
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        if terminated:
            success += 1
        rewards.append(ep_reward)
        steps_list.append(step + 1)
    return success, sum(rewards) / len(rewards), sum(steps_list) / len(steps_list)


def main():
    env = DoorKeyEnv(seed=ENV_SEED)

    if QTABLE_PATH.exists():
        print(f"[load] Q-tabla desde {QTABLE_PATH.relative_to(ROOT)}")
        agent = QLearningAgent.load(QTABLE_PATH)
    else:
        print("[train] no hay Q-tabla guardada; entrenando desde cero")
        agent = train(env)
        QTABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        agent.save(QTABLE_PATH)
        print(f"[save] {QTABLE_PATH.relative_to(ROOT)}")

    print()
    print(f"Q-tabla tiene {len(agent.q)} estados aprendidos.")
    print()

    print("Evaluación con distintas estrategias (100 episodios cada una):")
    print(f"  {'estrategia':<35} {'éxito':>6} {'reward':>10} {'pasos':>7}")
    print(f"  {'-'*35} {'-'*6} {'-'*10} {'-'*7}")

    strategies = [
        ("greedy puro (rng_seed=42)",   0.0,  42),
        ("greedy puro (rng_seed=0)",    0.0,   0),
        ("greedy puro (rng_seed=1)",    0.0,   1),
        ("greedy puro (rng_seed=7)",    0.0,   7),
        ("greedy puro (rng_seed=123)",  0.0, 123),
        ("cuasi-greedy eps=0.01",       0.01, 42),
        ("cuasi-greedy eps=0.05",       0.05, 42),
        ("cuasi-greedy eps=0.10",       0.10, 42),
    ]

    for name, eps, seed in strategies:
        succ, rew, steps = eval_strategy(env, agent, eps=eps, rng_seed=seed)
        print(f"  {name:<35} {succ:>4}/100 {rew:>+10.3f} {steps:>7.1f}")


if __name__ == "__main__":
    main()
