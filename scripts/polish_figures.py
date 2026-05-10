"""Regenera artefactos pulidos para README y docs/final.tex.

Outputs:
  results/minigrid/episode_greedy.gif       (alta resolución, tile_size=64)
  results/minigrid/episode_greedy.mp4       (idem)
  results/minigrid/learning_curve.png       (curva con sombreado de 3 fases)
  results/maze/learning_curve.png           (curva pulida, mismo estilo)
  docs/figures/final_minigrid_greedy_frames.png  (frames en eventos semánticos)
  docs/figures/final_minigrid_learning_curve.png (idem learning_curve)
  docs/figures/final_maze_learning_curve.png     (idem maze curve)

Reusa la Q-tabla persistida (no re-entrena MiniGrid). Para el laberinto,
re-entrena rápido in-place (2K episodios, ~30s) para tener la curva por
episodio.
"""

from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import QLearningAgent  # noqa: E402
from src.minigrid.env import DoorKeyEnv  # noqa: E402
from src.maze.env import Maze, MazeEnv  # noqa: E402


RESULTS_MG = ROOT / "results" / "minigrid"
RESULTS_MZ = ROOT / "results" / "maze"
DOCS_FIG = ROOT / "docs" / "figures"

REWARD_COLOR = "#1565c0"
REWARD_COLOR_DARK = "#0d47a1"
EPSILON_COLOR = "#c62828"
MAX_REWARD_MG = 2.80
MAX_REWARD_MZ_DERIVED = 76.0  # +100 - 24 step costs (camino BFS de 25 celdas, 24 transiciones)


# =====================================================================
# MiniGrid: rollout greedy con detección de eventos + GIF HD
# =====================================================================

def minigrid_greedy_rollout(tile_size: int = 64):
    """Carga la Q-tabla, ejecuta un episodio greedy, devuelve:
      frames: lista de arrays RGB (uno por t=0..N)
      events: dict subtask_name -> step donde se levantó por primera vez

    El env se reinstancia con tile_size grande para renders nítidos.
    """
    import gymnasium as gym
    import minigrid  # noqa: F401

    env = DoorKeyEnv(seed=0)
    # subir resolución del render del env subyacente
    env.env.unwrapped.tile_size = tile_size

    agent = QLearningAgent.load(RESULTS_MG / "qtable.pkl")

    random.seed(42)
    state, _ = env.reset()
    frames = [env.render()]
    events: dict[str, int] = {}

    # Estado inicial: registrar flags ya activas si las hubiera (no debería)
    flag_keys = ["ball_moved", "key_picked", "door_open", "key_dropped", "box_picked"]
    flag_idx = {"ball_moved": 3, "key_picked": 4, "door_open": 5, "key_dropped": 6, "box_picked": 7}

    def flags_of(s):
        return {k: bool(s[flag_idx[k]]) for k in flag_keys}

    prev_flags = flags_of(state)

    for t in range(1, 600):
        action = agent.greedy_action(state)
        state, _, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        new_flags = flags_of(state)
        for k in flag_keys:
            if new_flags[k] and not prev_flags[k] and k not in events:
                events[k] = t
        prev_flags = new_flags
        if terminated or truncated:
            break

    env.close()
    return frames, events


def write_gif(frames: list[np.ndarray], path: Path, duration_ms: int = 250) -> None:
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"  wrote {path.relative_to(ROOT)} ({len(frames)} frames, {pil_frames[0].size})")


def write_mp4(frames: list[np.ndarray], path: Path, fps: int = 5,
              freeze_last_seconds: float = 2.0) -> None:
    n_freeze = int(round(freeze_last_seconds * fps))
    frames_ext = frames + [frames[-1]] * n_freeze
    h, w = frames_ext[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for f in frames_ext:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  wrote {path.relative_to(ROOT)} ({len(frames_ext)} frames @ {fps} fps, {w}x{h})")


# =====================================================================
# MiniGrid: frames en eventos semánticos
# =====================================================================

def minigrid_event_frames(frames: list[np.ndarray], events: dict[str, int],
                         out_path: Path) -> None:
    """Selecciona 6 frames de eventos clave + título del subtask en cada uno."""
    last_t = len(frames) - 1

    # Frame 0: estado inicial. Resto: justo después de cada subtask.
    plan = [
        (0,                      "inicio"),
        (events["ball_moved"],   "bola movida"),
        (events["key_picked"],   "llave en mano"),
        (events["door_open"],    "puerta abierta"),
        (events["key_dropped"],  "llave soltada"),
        (last_t,                 "caja recogida"),
    ]

    # Validación: faltan eventos
    missing = [name for t, name in plan if t is None or (isinstance(t, int) and t < 0)]
    if missing:
        raise RuntimeError(f"Eventos no detectados: {missing}; events={events}")

    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.0 * rows))
    for ax, (t, label) in zip(axes.flat, plan):
        ax.imshow(frames[t])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"$t={t}$ — {label}", fontsize=11)
        for spine in ax.spines.values():
            spine.set_edgecolor("#999")
            spine.set_linewidth(0.8)
    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(ROOT)}")


# =====================================================================
# MiniGrid: curva con sombreado de 3 fases (datos de checkpoint)
# =====================================================================

# Progresión real del entrenamiento (verify_minigrid.py output)
MINIGRID_PROGRESSION = [
    # (ep, epsilon, avg_reward_last_500)
    (500,    0.779, +0.201),
    (1000,   0.606, +1.300),
    (1500,   0.472, +1.812),
    (2000,   0.368, +1.531),
    (2500,   0.286, +1.737),
    (3000,   0.223, +1.651),
    (3500,   0.174, +1.477),
    (4000,   0.135, +1.300),
    (5000,   0.082, +1.118),
    (7500,   0.050, +2.464),
    (10000,  0.050, +2.722),
]

MINIGRID_PHASES = [
    (0,     1500,  "Subida inicial",    "#fff3e0"),
    (1500,  5500,  "Plateau volátil",   "#fbe9e7"),
    (5500,  10500, "Convergencia",      "#e8f5e9"),
]


def minigrid_learning_curve(out_path: Path, *, title: str | None = None) -> None:
    eps_x = np.array([r[0] for r in MINIGRID_PROGRESSION])
    eps_y = np.array([r[1] for r in MINIGRID_PROGRESSION])
    rew_y = np.array([r[2] for r in MINIGRID_PROGRESSION])

    fig, ax = plt.subplots(figsize=(8.6, 4.4))

    # Sombreado de fases
    for lo, hi, label, color in MINIGRID_PHASES:
        ax.axvspan(lo, hi, color=color, alpha=0.85, zorder=0)
        center = (lo + min(hi, 10500)) / 2
        ax.text(center, 3.12, label,
                ha="center", va="bottom", fontsize=9, color="#444", style="italic")

    # Máximo teórico
    ax.axhline(MAX_REWARD_MG, color="#888", linestyle="--", linewidth=0.9, alpha=0.85, zorder=1)
    ax.text(10300, MAX_REWARD_MG + 0.07, f"máx. teórico (+{MAX_REWARD_MG:.2f})",
            ha="right", va="bottom", fontsize=8.5, color="#666")

    # Recompensa
    line_r, = ax.plot(eps_x, rew_y, marker="o", markersize=5.5,
                     color=REWARD_COLOR, linewidth=2.0,
                     label="Recompensa promedio (últ. 500 ep)", zorder=3)

    ax.set_xlabel("Episodio")
    ax.set_ylabel("Recompensa", color=REWARD_COLOR_DARK)
    ax.tick_params(axis="y", labelcolor=REWARD_COLOR_DARK)
    ax.set_ylim(-0.4, 3.3)
    ax.set_xlim(0, 10500)
    ax.grid(True, axis="both", alpha=0.25, zorder=1)

    # Epsilon en eje derecho
    ax2 = ax.twinx()
    line_e, = ax2.plot(eps_x, eps_y, marker="s", markersize=4.5,
                       color=EPSILON_COLOR, linewidth=1.4,
                       linestyle="--", alpha=0.9,
                       label=r"$\varepsilon$ (exploración)", zorder=2)
    ax2.set_ylabel(r"$\varepsilon$", color=EPSILON_COLOR)
    ax2.tick_params(axis="y", labelcolor=EPSILON_COLOR)
    ax2.set_ylim(0, 1.0)

    ax.legend([line_r, line_e], [line_r.get_label(), line_e.get_label()],
              loc="center right", fontsize=9, framealpha=0.95)

    if title:
        ax.set_title(title, fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(ROOT)}")


# =====================================================================
# Laberinto: re-entrenamiento corto + curva pulida
# =====================================================================

MAZE_PHASES = [
    (0,     300,   "Caos inicial",        "#fff3e0"),
    (300,   1000,  "Aprendizaje rápido",  "#fbe9e7"),
    (1000,  2000,  "Convergencia",        "#e8f5e9"),
]


def maze_train_for_curve(n_episodes: int = 2000, max_steps: int = 200):
    random.seed(42)
    maze = Maze.from_file(ROOT / "data" / "project_lab_v2.txt")
    env = MazeEnv(maze, max_steps=max_steps)
    agent = QLearningAgent(
        n_actions=env.n_actions,
        alpha=0.1, gamma=0.99,
        epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.999,
    )

    rewards, steps = [], []
    epsilons = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        for t in range(max_steps):
            action = agent.select_action(state)
            next_state, r, terminated, truncated, _ = env.step(action)
            agent.update(state, action, r, next_state, done=terminated)
            state = next_state
            ep_reward += r
            if terminated or truncated:
                break
        agent.decay_epsilon()
        rewards.append(ep_reward)
        steps.append(t + 1)
        epsilons.append(agent.epsilon)
    return np.array(rewards), np.array(steps), np.array(epsilons)


def maze_learning_curve(out_path: Path) -> None:
    print("  [maze] entrenando 2000 episodios para regenerar curva...")
    rewards, steps, epsilons = maze_train_for_curve()

    window = 50
    rew_smooth = np.array([rewards[max(0, i - window + 1):i + 1].mean()
                          for i in range(len(rewards))])
    steps_smooth = np.array([steps[max(0, i - window + 1):i + 1].mean()
                            for i in range(len(steps))])

    fig, (ax_r, ax_s) = plt.subplots(1, 2, figsize=(12, 4.2),
                                     gridspec_kw=dict(wspace=0.28))

    # Sombreado sin etiquetas inline (fase 1 es muy estrecha para
    # caber el texto sin traslaparse con los datos). Los colores los
    # explica el caption.
    def shade_phases(ax):
        for lo, hi, _, color in MAZE_PHASES:
            ax.axvspan(lo, hi, color=color, alpha=0.85, zorder=0)

    # Panel izquierdo: recompensa
    shade_phases(ax_r)
    ax_r.plot(rewards, alpha=0.22, color=REWARD_COLOR, linewidth=0.6,
              label="por episodio", zorder=2)
    ax_r.plot(rew_smooth, color=REWARD_COLOR_DARK, linewidth=1.8,
              label=f"media móvil ({window} ep)", zorder=3)
    ax_r.axhline(MAX_REWARD_MZ_DERIVED, color="#888", linestyle="--",
                 linewidth=0.9, alpha=0.85, zorder=1)
    ax_r.text(1980, MAX_REWARD_MZ_DERIVED + 3,
              f"óptimo BFS (+{int(MAX_REWARD_MZ_DERIVED)})",
              ha="right", va="bottom", fontsize=8.5, color="#666")
    ax_r.set_xlabel("Episodio")
    ax_r.set_ylabel("Recompensa")
    ax_r.set_xlim(0, 2000)
    ax_r.set_ylim(-220, 130)
    ax_r.grid(True, alpha=0.25, zorder=1)
    ax_r.legend(loc="lower right", fontsize=9, framealpha=0.95)

    # Panel derecho: pasos
    shade_phases(ax_s)
    ax_s.plot(steps, alpha=0.22, color="#37474f", linewidth=0.6,
              label="por episodio", zorder=2)
    ax_s.plot(steps_smooth, color="#263238", linewidth=1.8,
              label=f"media móvil ({window} ep)", zorder=3)
    ax_s.axhline(25, color="#2e7d32", linestyle="--",
                 linewidth=1.0, alpha=0.95, zorder=1)
    # Texto del BFS optimo en zona libre del panel (centro-izquierda, baja)
    ax_s.text(600, 33, "óptimo BFS (25 pasos)",
              ha="left", va="bottom", fontsize=8.5, color="#2e7d32",
              bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    ax_s.set_xlabel("Episodio")
    ax_s.set_ylabel("Pasos hasta la meta")
    ax_s.set_xlim(0, 2000)
    ax_s.set_ylim(0, 240)
    ax_s.grid(True, alpha=0.25, zorder=1)
    ax_s.legend(loc="upper right", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(ROOT)}")


# =====================================================================
# Main
# =====================================================================

def main():
    print("[1/4] MiniGrid greedy rollout (tile_size=64)")
    frames, events = minigrid_greedy_rollout(tile_size=64)
    print(f"      rollout: {len(frames)} frames; eventos: {events}")

    print("[2/4] Regenerando GIF y MP4 de MiniGrid en alta resolución")
    write_gif(frames, RESULTS_MG / "episode_greedy.gif", duration_ms=250)
    write_mp4(frames, RESULTS_MG / "episode_greedy.mp4", fps=5)

    print("[3/4] Figura de fotogramas semánticos")
    minigrid_event_frames(frames, events, DOCS_FIG / "final_minigrid_greedy_frames.png")

    print("[4/4] Curvas de aprendizaje con sombreado de fases")
    minigrid_learning_curve(RESULTS_MG / "learning_curve.png")
    minigrid_learning_curve(DOCS_FIG / "final_minigrid_learning_curve.png")
    maze_learning_curve(RESULTS_MZ / "learning_curve.png")
    maze_learning_curve(DOCS_FIG / "final_maze_learning_curve.png")

    print("\nListo.")


if __name__ == "__main__":
    main()
