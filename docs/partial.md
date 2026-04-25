# Entrega parcial — Caracterización del problema

## Descripción del ambiente

El ambiente es un laberinto rectangular de **8 filas × 7 columnas (56 celdas)** con muros internos que bloquean transiciones entre celdas adyacentes. La geometría se carga desde el archivo `data/project_lab_v2.txt` provisto en el enunciado, que define:

- Línea 1: dimensiones `nrows ncols = 8 7`.
- Línea 2: número de muros `nwalls = 65`.
- Líneas siguientes: cada una un segmento de muro `x1 y1 x2 y2` en el espacio de esquinas, donde `X` es fila (vertical, 0 arriba) y `Y` es columna (horizontal, 0 a la izquierda) — convención idéntica a la imagen del enunciado.

De los 65 segmentos, **37 son muros internos** (separan dos celdas válidas) y los 28 restantes son borde exterior, que el chequeo de límites ya cubre.

| Elemento     | Valor    |
|--------------|----------|
| Celda inicial (start) | `(6, 0)` |
| Celda meta (goal)     | `(1, 6)` |
| Camino óptimo (BFS)   | 25 pasos |

El agente debe llegar de `start` a `goal` minimizando el número de pasos. El ambiente es **determinista**: una misma acción desde el mismo estado siempre produce la misma transición.

## 1. Conjunto de estados

El estado se codifica como una tupla hashable `(row, col)` apta para indexar la Q-tabla:

| Componente | Tipo  | Rango     | Descripción                                    |
|------------|-------|-----------|------------------------------------------------|
| `row`      | `int` | `0..7`    | Fila de la celda donde está el agente          |
| `col`      | `int` | `0..6`    | Columna de la celda donde está el agente       |

- **Tamaño total del espacio de estados**: `|S| = 8 × 7 = 56`.
- **Estado inicial**: `s₀ = (6, 0)` (fijo en cada episodio).
- **Estado terminal**: `s_T = (1, 6)` (alcanzar esta celda termina el episodio con éxito).
- **Otras características**:
  - El estado captura toda la información Markoviana relevante: el agente no necesita recordar la trayectoria pasada porque la geometría del laberinto y la regla de transición dependen solo de `(row, col)` y de la acción.
  - No hay observabilidad parcial: el agente conoce su posición exacta.

## 2. Conjunto de acciones

El agente dispone de **4 acciones discretas** correspondientes a movimientos en la grilla a 4-conexa:

| Índice | Nombre  | Δ (fila, col) | Aplicabilidad |
|--------|---------|---------------|---------------|
| 0      | `UP`    | `(-1, 0)`     | Siempre ejecutable |
| 1      | `DOWN`  | `(+1, 0)`     | Siempre ejecutable |
| 2      | `LEFT`  | `(0, -1)`     | Siempre ejecutable |
| 3      | `RIGHT` | `(0, +1)`     | Siempre ejecutable |

**Semántica de "aplicabilidad"**: las cuatro acciones están disponibles desde cualquier estado. Lo que cambia es el efecto de la acción:

- Si la celda destino (`s + Δa`) está **dentro de los límites** del laberinto **y** no hay muro entre la celda origen `s` y la celda destino, la transición se realiza: `s' = s + Δa`.
- En caso contrario (intento de salir del grid o muro bloqueante), el agente **permanece en `s`** (`s' = s`). Llamamos a esto un *bump*: la acción se "ejecutó" pero no produjo desplazamiento.

Esta política — acciones siempre disponibles, transiciones inválidas mapeadas a no-op — simplifica la Q-tabla (no hay máscaras de acción por estado) y es estándar en gridworlds de RL.

## 3. Función de recompensa

La recompensa se modela como `r(s, a, s')` (depende de la transición). Sólo dos situaciones son distintas:

| Situación                                    | Recompensa | Comentario |
|----------------------------------------------|-----------:|------------|
| `s' = goal` (la transición lleva a la meta)  |   **+100** | Episodio termina exitosamente |
| `s' ≠ goal` (cualquier otra transición)      |    **−1**  | Costo de paso. Aplica también al *bump* (la celda destino bloqueada deja `s' = s`, sigue costando −1) |

**Pareja estado-acción explícita** (forma equivalente para la rúbrica del enunciado):

| `(s, a)`                                                        | `s'`                  | `r` |
|-----------------------------------------------------------------|-----------------------|----:|
| `s` adyacente al goal con `a` orientada hacia el goal           | `goal`                |  +100 |
| `s` con `a` que produce desplazamiento válido (no es al goal)   | `s + Δa`              |  −1 |
| `s` con `a` que sale del grid o cruza un muro                   | `s` (bump)            |  −1 |

### Justificación del diseño

- **Costo de paso negativo**: la única forma de maximizar el retorno acumulado es minimizar la cantidad de pasos. Sin costo de paso, todas las políticas que eventualmente llegan al goal serían óptimas; con `-1` por paso, sólo lo es la trayectoria más corta.
- **Recompensa terminal grande (+100) frente al costo de paso (-1)**: garantiza que la señal terminal domina sobre los costos acumulados (incluso un episodio que rebote 99 veces antes de llegar al goal recibe retorno positivo, lo que evita que el agente "evite" llegar). Para el camino óptimo de 25 pasos el retorno es `24·(-1) + 100 = +76`; el peor caso truncado a 200 pasos da `-200`.
- **Choque sin penalización adicional**: el costo de paso ya penaliza implícitamente al *bump* (es un paso desperdiciado). Agregar una penalización extra aceleraría el aprendizaje pero complicaría el modelo de recompensa sin necesidad para un ambiente de este tamaño (56 estados convergen rápido).
- **Ningún reward shaping intermedio** (por ejemplo, bonificación por acercarse a la meta): la formulación es la canónica de gridworld con costo uniforme de paso, lo que mantiene la política óptima alineada exactamente con el camino BFS más corto y facilita la verificación.

### Modelo de transición y dinámica del episodio

- **Transición**: determinista. `T(s' | s, a) = 1` para el `s'` definido por las reglas anteriores, `0` para todos los demás.
- **Terminación**: el episodio termina (`terminated = True`) al alcanzar `goal`. Si se exceden 200 pasos sin llegar, se trunca (`truncated = True`); la truncación **no** corta el bootstrap de Q-learning porque no es un estado terminal real.
- **Factor de descuento sugerido**: `γ = 0.99` para que el bonus terminal mantenga influencia hacia atrás a lo largo de los 25 pasos del camino óptimo (`0.99²⁴ ≈ 0.79`).

## Referencias visuales

- `notebooks/01_exploration.ipynb` carga el laberinto y reproduce la figura del enunciado, además de calcular y dibujar el camino óptimo BFS (referencia de evaluación).
- `notebooks/02_experiments.ipynb` entrena el agente y produce las gráficas de aprendizaje y el GIF del comportamiento aprendido.
