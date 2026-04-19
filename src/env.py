"""Ambiente DoorKey con bola bloqueando la puerta.

Envuelve un entorno de MiniGrid y expone una API discreta apropiada
para Q-learning tabular: estado codificado como tupla hashable,
conjunto fijo de acciones, recompensa con shaping por subtareas.
"""

from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401  # registra los entornos MiniGrid al importar


# Acciones de MiniGrid que usa el agente
ACTIONS = {
    0: "left",      # girar a la izquierda
    1: "right",     # girar a la derecha
    2: "forward",   # avanzar
    3: "pickup",    # recoger objeto (llave o bola)
    4: "drop",      # soltar objeto
    5: "toggle",    # abrir/cerrar puerta
}


class DoorKeyEnv:
    def __init__(self, env_id: str = "MiniGrid-BlockedUnlockPickup-v0", seed: int | None = None):
        self.env = gym.make(env_id, render_mode="rgb_array")
        self.seed = seed
        self._subtasks = {"has_key": False, "ball_moved": False, "door_open": False}
        self.reset()

    def reset(self):
        obs, info = self.env.reset(seed=self.seed)
        self._subtasks = {"has_key": False, "ball_moved": False, "door_open": False}
        self._initial_ball_pos = self._find_object("ball")
        return self.get_state(), info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = self._shape_reward(reward, terminated)
        return self.get_state(), shaped, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    # ------- Estado -------

    def get_state(self) -> tuple:
        """Codifica el estado como tupla hashable para indexar la Q-tabla."""
        unwrapped = self.env.unwrapped
        pos = tuple(unwrapped.agent_pos)
        direction = int(unwrapped.agent_dir)
        has_key = self._has_object("key")
        ball_moved = self._ball_has_moved()
        door_open = self._is_door_open()
        self._subtasks = {"has_key": has_key, "ball_moved": ball_moved, "door_open": door_open}
        return (pos, direction, has_key, ball_moved, door_open)

    @property
    def n_actions(self) -> int:
        return len(ACTIONS)

    # ------- Shaping de recompensa -------

    def _shape_reward(self, base_reward: float, terminated: bool) -> float:
        reward = -0.01  # costo por paso
        if terminated and base_reward > 0:
            reward += 1.0
        # bonificaciones por subtareas (una sola vez)
        new_has_key = self._has_object("key")
        new_ball_moved = self._ball_has_moved()
        new_door_open = self._is_door_open()
        if new_has_key and not self._subtasks["has_key"]:
            reward += 0.3
        if new_ball_moved and not self._subtasks["ball_moved"]:
            reward += 0.2
        if new_door_open and not self._subtasks["door_open"]:
            reward += 0.5
        return reward

    # ------- Utilidades internas -------

    def _has_object(self, obj_type: str) -> bool:
        carrying = self.env.unwrapped.carrying
        return carrying is not None and carrying.type == obj_type

    def _find_object(self, obj_type: str):
        grid = self.env.unwrapped.grid
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is not None and cell.type == obj_type:
                    return (x, y)
        return None

    def _ball_has_moved(self) -> bool:
        if self._initial_ball_pos is None:
            return True
        current = self._find_object("ball")
        carrying_ball = self._has_object("ball")
        return carrying_ball or current != self._initial_ball_pos

    def _is_door_open(self) -> bool:
        grid = self.env.unwrapped.grid
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is not None and cell.type == "door":
                    return bool(getattr(cell, "is_open", False))
        return False
