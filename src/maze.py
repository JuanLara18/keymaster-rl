"""Laberinto del proyecto y ambiente Q-learning asociado.

El laberinto se carga desde un archivo de texto con el formato del
enunciado (`data/project_lab_v2.txt`). Las celdas se indexan por
`(row, col)` con `row ∈ [0, nrows)` y `col ∈ [0, ncols)`. Los muros se
describen como segmentos unitarios en el espacio de esquinas: cada muro
separa exactamente dos celdas adyacentes (o una celda interior del
exterior, en cuyo caso es borde y se ignora porque la verificación de
límites ya lo cubre).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------- Acciones ----------

ACTIONS = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


# ---------- Laberinto ----------

class Maze:
    """Geometría del laberinto: dimensiones, muros, celdas inicial y meta."""

    def __init__(
        self,
        nrows: int,
        ncols: int,
        walls: set[frozenset[tuple[int, int]]],
        wall_segments: list[tuple[int, int, int, int]],
        start: tuple[int, int] = (6, 0),
        goal: tuple[int, int] = (1, 6),
    ):
        self.nrows = nrows
        self.ncols = ncols
        self.walls = walls               # solo muros internos (entre dos celdas válidas)
        self.wall_segments = wall_segments  # todos los segmentos del archivo (para render)
        self.start = start
        self.goal = goal
        self._validate()

    # ------- Construcción -------

    @classmethod
    def from_file(cls, path: str | Path, start=(6, 0), goal=(1, 6)) -> "Maze":
        lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
        nrows, ncols = (int(x) for x in lines[0].split())
        nwalls = int(lines[1])
        if len(lines) - 2 != nwalls:
            raise ValueError(
                f"Cabecera dice {nwalls} muros pero el archivo trae {len(lines) - 2}"
            )

        wall_segments: list[tuple[int, int, int, int]] = []
        walls: set[frozenset[tuple[int, int]]] = set()
        for raw in lines[2:]:
            x1, y1, x2, y2 = (int(v) for v in raw.split())
            wall_segments.append((x1, y1, x2, y2))
            cells = cls._cells_separated_by(x1, y1, x2, y2, nrows, ncols)
            if cells is not None:
                walls.add(frozenset(cells))
        return cls(nrows, ncols, walls, wall_segments, start, goal)

    @staticmethod
    def _cells_separated_by(
        x1: int, y1: int, x2: int, y2: int, nrows: int, ncols: int
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """Devuelve las dos celdas que un segmento de muro separa, o None si es borde.

        Convención (X = fila, Y = columna; igual a la imagen del enunciado):
        - x1 == x2: segmento horizontal en la imagen → separa (x1-1, y1) y (x1, y1).
        - y1 == y2: segmento vertical en la imagen   → separa (x1, y1-1) y (x1, y1).
        """
        if x1 == x2:
            row_top, row_bot, col = x1 - 1, x1, y1
            a, b = (row_top, col), (row_bot, col)
        elif y1 == y2:
            row, col_left, col_right = x1, y1 - 1, y1
            a, b = (row, col_left), (row, col_right)
        else:
            raise ValueError(f"Muro no axis-aligned: ({x1},{y1})-({x2},{y2})")

        def in_grid(cell):
            r, c = cell
            return 0 <= r < nrows and 0 <= c < ncols

        if in_grid(a) and in_grid(b):
            return (a, b)
        return None

    # ------- API de consulta -------

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < self.nrows and 0 <= c < self.ncols

    def is_blocked(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        """True si moverse de a a b está bloqueado (fuera del grid o por muro)."""
        if not self.in_bounds(a) or not self.in_bounds(b):
            return True
        return frozenset((a, b)) in self.walls

    def neighbors(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        """Vecinos accesibles a 4-conexa (sin muro y dentro del grid)."""
        r, c = cell
        out = []
        for dr, dc in DELTAS.values():
            nb = (r + dr, c + dc)
            if self.in_bounds(nb) and not self.is_blocked(cell, nb):
                out.append(nb)
        return out

    def shortest_path(self) -> list[tuple[int, int]] | None:
        """BFS start→goal. Devuelve lista de celdas (incluye start y goal) o None."""
        q = deque([(self.start, [self.start])])
        seen = {self.start}
        while q:
            cell, path = q.popleft()
            if cell == self.goal:
                return path
            for nb in self.neighbors(cell):
                if nb not in seen:
                    seen.add(nb)
                    q.append((nb, path + [nb]))
        return None

    def n_states(self) -> int:
        return self.nrows * self.ncols

    # ------- Render -------

    def render(
        self,
        ax=None,
        agent: tuple[int, int] | None = None,
        path: Iterable[tuple[int, int]] | None = None,
        title: str | None = None,
    ):
        """Dibuja el laberinto reproduciendo la convención de la imagen del enunciado.

        Eje vertical = filas (X), 0 arriba; eje horizontal = columnas (Y).
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 7))

        # Muros: se dibujan TODOS los segmentos del archivo (incluye bordes).
        for x1, y1, x2, y2 in self.wall_segments:
            ax.plot([y1, y2], [x1, x2], color="black", linewidth=2.5)

        # Camino opcional
        if path is not None:
            path = list(path)
            xs = [c + 0.5 for r, c in path]
            ys = [r + 0.5 for r, c in path]
            ax.plot(xs, ys, color="tab:green", linewidth=2, alpha=0.7,
                    marker="o", markersize=5)

        # Start y goal
        sr, sc = self.start
        gr, gc = self.goal
        ax.plot(sc + 0.5, sr + 0.5, "o", color="red", markersize=18, label="start")
        ax.plot(gc + 0.5, gr + 0.5, "o", color="blue", markersize=18, label="goal")

        # Agente (sobreescribe cualquier marker debajo)
        if agent is not None:
            ar, ac = agent
            ax.plot(ac + 0.5, ar + 0.5, "o", color="orange",
                    markersize=14, label="agente")

        # Grid sutil
        for r in range(self.nrows + 1):
            ax.axhline(r, color="lightgray", linewidth=0.5, zorder=0)
        for c in range(self.ncols + 1):
            ax.axvline(c, color="lightgray", linewidth=0.5, zorder=0)

        ax.set_xlim(0, self.ncols)
        ax.set_ylim(self.nrows, 0)  # invertido: row 0 arriba
        ax.set_aspect("equal")
        ax.set_xlabel("Columnas (Y)")
        ax.set_ylabel("Filas (X)")
        ax.set_xticks(range(self.ncols + 1))
        ax.set_yticks(range(self.nrows + 1))
        if title:
            ax.set_title(title)
        return ax

    # ------- Validación interna -------

    def _validate(self):
        if not self.in_bounds(self.start):
            raise ValueError(f"start {self.start} fuera del grid {self.nrows}x{self.ncols}")
        if not self.in_bounds(self.goal):
            raise ValueError(f"goal {self.goal} fuera del grid {self.nrows}x{self.ncols}")
        if self.shortest_path() is None:
            raise ValueError("No existe camino entre start y goal — laberinto inconsistente")


# ---------- Ambiente ----------

class MazeEnv:
    """Ambiente determinista para Q-learning tabular.

    API mínima estilo gym:
        reset()  -> (state, info)
        step(a)  -> (state, reward, terminated, truncated, info)

    Recompensa:
        +100 al transicionar a la celda meta (terminated=True).
        -1   en cualquier otra transición (incluido un choque que deje al
             agente en la misma celda).

    El agente arranca siempre en `maze.start` y termina al llegar a
    `maze.goal` o al alcanzar `max_steps` (truncated=True).
    """

    def __init__(self, maze: Maze, max_steps: int = 200):
        self.maze = maze
        self.max_steps = int(max_steps)
        self.state = maze.start
        self.steps = 0

    @property
    def n_actions(self) -> int:
        return 4

    def reset(self) -> tuple[tuple[int, int], dict]:
        self.state = self.maze.start
        self.steps = 0
        return self.state, {}

    def step(self, action: int) -> tuple[tuple[int, int], float, bool, bool, dict]:
        if action not in DELTAS:
            raise ValueError(f"acción inválida: {action}")

        dr, dc = DELTAS[action]
        proposed = (self.state[0] + dr, self.state[1] + dc)
        bumped = (
            not self.maze.in_bounds(proposed)
            or self.maze.is_blocked(self.state, proposed)
        )
        new_state = self.state if bumped else proposed

        self.state = new_state
        self.steps += 1
        terminated = (self.state == self.maze.goal)
        truncated = (not terminated) and (self.steps >= self.max_steps)
        reward = 100.0 if terminated else -1.0
        return self.state, reward, terminated, truncated, {"bumped": bumped}
