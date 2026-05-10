"""Genera figuras para docs/partial.tex.

Salidas:
  docs/figures/minigrid_learning_curve.png
  docs/figures/minigrid_greedy_frames.png

Datos fuente:
  - Tabla discreta de avg-reward y epsilon por episodio (ver PROGRESSION).
  - GIF del agente greedy: results/minigrid/episode_greedy.gif.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
GIF_PATH = ROOT / "results" / "minigrid" / "episode_greedy.gif"

# Tabla de progresión (ep, epsilon, avg_reward) tomada de _archive/docs/final.md.
PROGRESSION = [
    (500, 0.779, -0.015),
    (1000, 0.606, -0.090),
    (1500, 0.472, -0.093),
    (2000, 0.368, 0.131),
    (2500, 0.286, 0.333),
    (3000, 0.223, 0.490),
    (3500, 0.174, 1.269),
    (4000, 0.135, 2.461),
    (5000, 0.082, 2.311),
    (7500, 0.050, 2.748),
    (10000, 0.050, 2.568),
]

# Fases observadas en la curva.
PHASES = [
    (0, 2000, "Exploración", "#f5f5f5"),
    (2000, 4000, "Aprendizaje rápido", "#fff3e0"),
    (4000, 10500, "Convergencia", "#e8f5e9"),
]

REWARD_COLOR = "#1565c0"
EPSILON_COLOR = "#c62828"
MAX_THEORETICAL = 2.80


def learning_curve(out_path: Path) -> None:
    eps_x = np.array([row[0] for row in PROGRESSION])
    eps_y = np.array([row[1] for row in PROGRESSION])
    rew_y = np.array([row[2] for row in PROGRESSION])

    fig, ax = plt.subplots(figsize=(8.4, 4.4))

    # Sombreado de las tres fases
    for lo, hi, label, color in PHASES:
        ax.axvspan(lo, hi, color=color, alpha=0.85, zorder=0)

    # Línea horizontal de máximo teórico
    ax.axhline(
        MAX_THEORETICAL,
        color="#888",
        linestyle="--",
        linewidth=0.9,
        alpha=0.85,
        zorder=1,
    )
    ax.text(
        10300,
        MAX_THEORETICAL + 0.07,
        f"máx. teórico (+{MAX_THEORETICAL:.2f})",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#666",
    )

    # Etiquetas de fase en la parte superior del área
    for lo, hi, label, _ in PHASES:
        center = (lo + min(hi, 10500)) / 2
        ax.text(
            center,
            3.08,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#444",
            style="italic",
        )

    # Recompensa (eje principal)
    line_r, = ax.plot(
        eps_x,
        rew_y,
        marker="o",
        markersize=5.5,
        color=REWARD_COLOR,
        linewidth=2.0,
        label="Recompensa promedio (últ. 500 ep)",
        zorder=3,
    )

    ax.set_xlabel("Episodio")
    ax.set_ylabel("Recompensa", color=REWARD_COLOR)
    ax.tick_params(axis="y", labelcolor=REWARD_COLOR)
    ax.set_ylim(-0.6, 3.25)
    ax.set_xlim(0, 10500)
    ax.grid(True, axis="both", alpha=0.25, zorder=1)

    # Epsilon (eje secundario)
    ax2 = ax.twinx()
    line_e, = ax2.plot(
        eps_x,
        eps_y,
        marker="s",
        markersize=4.5,
        color=EPSILON_COLOR,
        linewidth=1.4,
        linestyle="--",
        alpha=0.9,
        label=r"$\varepsilon$ (exploración)",
        zorder=2,
    )
    ax2.set_ylabel(r"$\varepsilon$", color=EPSILON_COLOR)
    ax2.tick_params(axis="y", labelcolor=EPSILON_COLOR)
    ax2.set_ylim(0, 1.0)

    # Leyenda combinada
    ax.legend(
        [line_r, line_e],
        [line_r.get_label(), line_e.get_label()],
        loc="center right",
        fontsize=9,
        framealpha=0.95,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(ROOT)}")


def greedy_frames(out_path: Path, n_frames: int = 6, rows: int = 2, cols: int = 3) -> None:
    if not GIF_PATH.exists():
        raise FileNotFoundError(f"GIF no encontrado: {GIF_PATH}")
    if rows * cols != n_frames:
        raise ValueError("rows * cols debe coincidir con n_frames")

    img = Image.open(GIF_PATH)
    total = getattr(img, "n_frames", 1)
    if total < 2:
        raise RuntimeError("El GIF tiene un solo frame; no hay nada que muestrear.")

    indices = np.linspace(0, total - 1, n_frames).round().astype(int).tolist()
    frames = []
    for idx in indices:
        img.seek(idx)
        frames.append(img.convert("RGB").copy())

    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.5 * rows))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    for ax, frame, idx in zip(axes_flat, frames, indices):
        ax.imshow(np.asarray(frame))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"$t = {idx}$", fontsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor("#999")
            spine.set_linewidth(0.8)

    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(ROOT)} (indices: {indices})")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[build_figures] writing into {FIG_DIR.relative_to(ROOT)}")
    learning_curve(FIG_DIR / "minigrid_learning_curve.png")
    greedy_frames(FIG_DIR / "minigrid_greedy_frames.png")


if __name__ == "__main__":
    main()
