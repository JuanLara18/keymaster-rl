"""Re-corre el entrenamiento de MiniGrid y guarda los artefactos:
  - results/minigrid/qtable.pkl
  - results/minigrid/learning_curve.png
  - results/minigrid/episode_greedy.gif

Reporta tabla de progresión, éxito greedy y estados visitados.
"""

from __future__ import annotations

import random
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import QLearningAgent  # noqa: E402
from src.minigrid.env import DoorKeyEnv  # noqa: E402


N_EPISODES = 10000
MAX_STEPS = 576
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995
SEED = 0
PYRANDOM_SEED = 42

CHECKPOINTS = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 7500, 10000]

RESULTS_DIR = ROOT / "results" / "minigrid"
QTABLE_PATH = RESULTS_DIR / "qtable.pkl"
CURVE_PATH = RESULTS_DIR / "learning_curve.png"
GIF_PATH = RESULTS_DIR / "episode_greedy.gif"


def train():
    random.seed(PYRANDOM_SEED)
    env = DoorKeyEnv(seed=SEED)
    agent = QLearningAgent(
        n_actions=env.n_actions,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
    )

    rewards_history = deque(maxlen=500)
    rewards_per_episode: list[float] = []
    epsilon_per_episode: list[float] = []
    progression: list[tuple[int, float, float]] = []  # (ep, eps, avg_reward)
    visited = set()

    t0 = time.time()
    for ep in range(1, N_EPISODES + 1):
        state, _ = env.reset()
        visited.add(state)
        ep_reward = 0.0
        for step in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            visited.add(next_state)
            agent.update(state, action, reward, next_state, done=terminated)
            state = next_state
            ep_reward += reward
            if terminated or truncated:
                break
        agent.decay_epsilon()
        rewards_history.append(ep_reward)
        rewards_per_episode.append(ep_reward)
        epsilon_per_episode.append(agent.epsilon)

        if ep in CHECKPOINTS:
            avg = sum(rewards_history) / len(rewards_history)
            progression.append((ep, agent.epsilon, avg))
            print(
                f"  ep={ep:>5}  eps={agent.epsilon:.3f}  "
                f"avg_last500={avg:+.3f}  visited={len(visited)}  "
                f"({time.time() - t0:.1f}s)"
            )

    return env, agent, progression, visited, rewards_per_episode, epsilon_per_episode


def save_learning_curve(rewards: list[float], epsilons: list[float], out_path: Path,
                        window: int = 500) -> None:
    arr = np.array(rewards)
    smoothed = np.array([
        arr[max(0, i - window + 1) : i + 1].mean() for i in range(len(arr))
    ])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(arr, alpha=0.18, color="#1565c0", linewidth=0.6, label="por episodio")
    ax.plot(smoothed, color="#0d47a1", linewidth=1.8,
            label=f"media móvil ({window} ep)")
    ax.axhline(2.80, linestyle="--", color="#777", linewidth=0.9, alpha=0.7)
    ax.text(len(arr) * 0.99, 2.83, "máx. teórico (+2.80)",
            ha="right", va="bottom", fontsize=8.5, color="#666")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Recompensa", color="#0d47a1")
    ax.tick_params(axis="y", labelcolor="#0d47a1")
    ax.set_xlim(0, len(arr))
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(epsilons, color="#c62828", linestyle="--", linewidth=1.2,
             alpha=0.85, label=r"$\varepsilon$")
    ax2.set_ylabel(r"$\varepsilon$", color="#c62828")
    ax2.tick_params(axis="y", labelcolor="#c62828")
    ax2.set_ylim(0, 1.0)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right",
              fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_greedy_gif(env: DoorKeyEnv, agent: QLearningAgent, out_path: Path,
                    duration_ms: int = 200) -> int:
    state, _ = env.reset()
    frames = [env.render()]
    for _ in range(MAX_STEPS):
        action = agent.greedy_action(state)
        state, _, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            break
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(out_path, save_all=True, append_images=pil_frames[1:],
                       duration=duration_ms, loop=0)
    return len(frames) - 1


def evaluate(env: DoorKeyEnv, agent: QLearningAgent, n_eval: int = 100):
    success = 0
    rewards = []
    steps_list = []
    for _ in range(n_eval):
        state, _ = env.reset()
        ep_reward = 0.0
        terminated = False
        for step in range(MAX_STEPS):
            action = agent.greedy_action(state)
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
    print("=" * 60)
    print("Entrenamiento MiniGrid-BlockedUnlockPickup-v0 (estado 8-tupla)")
    print("=" * 60)
    print(f"N_EPISODES={N_EPISODES}  MAX_STEPS={MAX_STEPS}")
    print(f"alpha={ALPHA}  gamma={GAMMA}  eps:{EPSILON}->{EPSILON_MIN} decay={EPSILON_DECAY}")
    print(f"env_seed={SEED}  random_seed={PYRANDOM_SEED}")
    print()

    print("Entrenamiento:")
    env, agent, progression, visited, rewards, epsilons = train()
    print()

    print(f"Estados visitados durante entrenamiento: {len(visited)}")
    print()

    print("Evaluación greedy (100 episodios):")
    success, avg_reward, avg_steps = evaluate(env, agent, n_eval=100)
    print(f"  éxito  = {success}/100")
    print(f"  reward = {avg_reward:+.3f}")
    print(f"  pasos  = {avg_steps:.1f}")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    agent.save(QTABLE_PATH)
    print(f"Q-tabla guardada en {QTABLE_PATH.relative_to(ROOT)}")

    save_learning_curve(rewards, epsilons, CURVE_PATH)
    print(f"Curva de aprendizaje guardada en {CURVE_PATH.relative_to(ROOT)}")

    n_steps_gif = save_greedy_gif(env, agent, GIF_PATH)
    print(f"GIF greedy ({n_steps_gif} pasos) guardado en {GIF_PATH.relative_to(ROOT)}")

    print()
    print("Tabla de progresión:")
    print(f"  {'ep':>5}  {'eps':>6}  {'avg_reward':>10}")
    for ep, eps, avg in progression:
        print(f"  {ep:>5}  {eps:>6.3f}  {avg:>+10.3f}")


if __name__ == "__main__":
    main()
