"""Ambiente DoorKey con bola bloqueando la puerta.

Envuelve `MiniGrid-BlockedUnlockPickup-v0` y expone una API discreta para
Q-learning tabular. El estado es la tupla de 8 componentes documentada en
`docs/partial.tex` (Sec. 3, ec. 1):

    s = (pos, direction, carrying, ball_moved,
         key_picked, door_open, key_dropped, box_picked)

Las tres últimas banderas (`key_picked`, `key_dropped`, `box_picked`) son
banderas de progreso del episodio: una vez pasan a True, no vuelven a
False. Forman parte del estado para que el shaping "primera vez" sea
Markoviano sobre `(s, a, s')`. Sin ellas, la misma transición producía
recompensas distintas según la historia previa del episodio.

Diseño:
    - `_achieved` es la fuente de verdad de las tres flags persistentes.
    - `get_state()` lee `_achieved` (no muta).
    - `_shape_reward()` muta `_achieved` antes de retornar la recompensa,
      de modo que la próxima llamada a `get_state()` ya vea las flags
      actualizadas.
"""

from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401  # registra los entornos MiniGrid al importar


# Acciones discretas de MiniGrid usadas por el agente.
ACTIONS = {
    0: "left",      # girar a la izquierda
    1: "right",     # girar a la derecha
    2: "forward",   # avanzar
    3: "pickup",    # recoger objeto
    4: "drop",      # soltar objeto
    5: "toggle",    # abrir/cerrar puerta
}

# Costo por paso. Pequeño (1e-3) frente al máximo de bonus (+2.80) para
# que un episodio largo (576 pasos × 1e-3 = 0.576) no domine al shaping.
STEP_COST = -0.001

# Bonificaciones por subtarea (una sola vez por episodio).
BONUS_BALL_MOVED = 0.20
BONUS_KEY_PICKED = 0.30
BONUS_DOOR_OPENED = 0.50
BONUS_KEY_DROPPED = 0.30  # solo después de abrir la puerta
BONUS_BOX_PICKED = 0.50
BONUS_GOAL = 1.00


class DoorKeyEnv:
    def __init__(
        self,
        env_id: str = "MiniGrid-BlockedUnlockPickup-v0",
        seed: int | None = None,
    ):
        self.env = gym.make(env_id, render_mode="rgb_array")
        self.seed = seed
        self._initial_ball_pos: tuple[int, int] | None = None
        self._achieved = self._fresh_achievements()
        self.reset()

    def reset(self):
        # Inicializar achievements ANTES del primer get_state(), porque
        # las flags forman parte del estado.
        self._achieved = self._fresh_achievements()
        obs, info = self.env.reset(seed=self.seed)
        self._initial_ball_pos = self._find_object("ball")
        return self.get_state(), info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # _shape_reward muta self._achieved; debe ejecutarse antes de
        # llamar a get_state() para que la nueva tupla refleje las flags
        # recién levantadas.
        shaped = self._shape_reward(reward, terminated)
        return self.get_state(), shaped, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    @property
    def n_actions(self) -> int:
        return len(ACTIONS)

    # ------- Estado -------

    def get_state(self) -> tuple:
        """Codifica el estado como tupla hashable de 8 componentes.

        Componentes:
            pos            -- (x, y) posición del agente.
            direction      -- 0..3 (E, S, W, N).
            carrying       -- "none" | "key" | "ball" | "box".
            ball_moved     -- True si la bola ya no está en su posición
                              inicial (o el agente la carga).
            key_picked     -- True si el agente recogió la llave en algún
                              momento del episodio (persistente).
            door_open      -- True si la puerta está abierta.
            key_dropped    -- True si el agente soltó la llave después de
                              abrir la puerta (persistente).
            box_picked     -- True si el agente cargó la caja al menos una
                              vez (persistente).
        """
        unwrapped = self.env.unwrapped
        pos = tuple(unwrapped.agent_pos)
        direction = int(unwrapped.agent_dir)
        return (
            pos,
            direction,
            self._carrying_type(),
            self._ball_has_moved(),
            self._achieved["key_picked"],
            self._is_door_open(),
            self._achieved["key_dropped"],
            self._achieved["box_picked"],
        )

    # ------- Shaping de recompensa -------

    def _shape_reward(self, base_reward: float, terminated: bool) -> float:
        reward = STEP_COST

        # Bonus terminal: MiniGrid devuelve base_reward > 0 únicamente al
        # recoger la caja objetivo (éxito).
        if terminated and base_reward > 0:
            reward += BONUS_GOAL

        carrying = self._carrying_type()
        ball_moved = self._ball_has_moved()
        door_open = self._is_door_open()

        if ball_moved and not self._achieved["ball_moved"]:
            reward += BONUS_BALL_MOVED
            self._achieved["ball_moved"] = True

        if carrying == "key" and not self._achieved["key_picked"]:
            reward += BONUS_KEY_PICKED
            self._achieved["key_picked"] = True

        if door_open and not self._achieved["door_opened"]:
            reward += BONUS_DOOR_OPENED
            self._achieved["door_opened"] = True

        # Soltar la llave solo cuenta DESPUÉS de abrir la puerta. Si el
        # agente la suelta antes (flujo subóptimo), no se le premia: el
        # objetivo del bonus es liberar el inventario para recoger la caja.
        if (
            self._achieved["door_opened"]
            and self._achieved["key_picked"]
            and carrying != "key"
            and not self._achieved["key_dropped"]
        ):
            reward += BONUS_KEY_DROPPED
            self._achieved["key_dropped"] = True

        if carrying == "box" and not self._achieved["box_picked"]:
            reward += BONUS_BOX_PICKED
            self._achieved["box_picked"] = True

        return reward

    # ------- Utilidades internas -------

    @staticmethod
    def _fresh_achievements() -> dict[str, bool]:
        return {
            "ball_moved": False,
            "key_picked": False,
            "door_opened": False,
            "key_dropped": False,
            "box_picked": False,
        }

    def _carrying_type(self) -> str:
        carrying = self.env.unwrapped.carrying
        return carrying.type if carrying is not None else "none"

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
        if self._carrying_type() == "ball":
            return True
        current = self._find_object("ball")
        return current != self._initial_ball_pos

    def _is_door_open(self) -> bool:
        grid = self.env.unwrapped.grid
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is not None and cell.type == "door":
                    return bool(getattr(cell, "is_open", False))
        return False
