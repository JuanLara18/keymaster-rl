# Entrega parcial — definición del ambiente

> Borrador v1. Por refinar con base en pruebas del notebook de exploración.

## Descripción del problema

El ambiente consiste en dos habitaciones separadas por una puerta con llave. El agente parte en la habitación izquierda y debe alcanzar una casilla objetivo en la habitación derecha. Para abrir la puerta necesita una llave del mismo color, pero antes de abrirla debe retirar una bola que la bloquea. Se usa como base `MiniGrid-BlockedUnlockPickup-v0` (por ajustar si se requiere un goal explícito).

## Estados

El estado se codifica como una tupla hashable apta para Q-learning tabular:

| Componente      | Tipo        | Rango                 | Descripción                         |
|-----------------|-------------|-----------------------|-------------------------------------|
| `pos`           | `(int, int)` | `(0..W-1, 0..H-1)`   | Posición del agente en la grilla    |
| `direction`     | `int`       | `0..3`                | Orientación (0=E, 1=S, 2=W, 3=N)    |
| `has_key`       | `bool`      | `{False, True}`       | El agente lleva la llave            |
| `ball_moved`    | `bool`      | `{False, True}`       | La bola ya no bloquea la puerta     |
| `door_open`     | `bool`      | `{False, True}`       | La puerta está abierta              |

## Acciones

| Índice | Nombre     | Aplicabilidad                                  |
|--------|------------|------------------------------------------------|
| 0      | `left`     | Siempre                                        |
| 1      | `right`    | Siempre                                        |
| 2      | `forward`  | Si la casilla de adelante es transitable       |
| 3      | `pickup`   | Si hay un objeto recogible enfrente            |
| 4      | `drop`     | Si el agente lleva un objeto y tiene espacio   |
| 5      | `toggle`   | Si hay puerta enfrente (abre si tiene llave)   |

## Función de recompensa

| Situación                                     | Recompensa |
|-----------------------------------------------|-----------:|
| Paso cualquiera                               |     −0.01  |
| Mover la bola por primera vez                 |     +0.20  |
| Recoger la llave por primera vez              |     +0.30  |
| Abrir la puerta por primera vez               |     +0.50  |
| Llegar al objetivo (episodio exitoso)         |     +1.00  |

Las bonificaciones por subtarea se entregan una sola vez por episodio, condicionadas a detectar la transición de estado correspondiente.
