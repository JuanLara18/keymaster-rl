# Entrega parcial — Caracterización del problema

> Enunciado vigente: post del profesor Nicolás Cardozo en el foro *Insumos proyecto* (18 de marzo, 2026), que actualiza la versión inicial del proyecto y entrega el archivo `data/project_lab_v2.txt` junto con la imagen de referencia del laberinto.

## Descripción del ambiente

Laberinto rectangular de **8 filas × 7 columnas (56 celdas)** con muros internos que bloquean transiciones entre celdas adyacentes. La geometría se carga desde `data/project_lab_v2.txt`, cuyo formato es:

- Línea 1: dimensiones $n_{\text{filas}}\, n_{\text{cols}} = 8\ 7$.
- Línea 2: número total de segmentos de muro $n_{\text{walls}} = 65$.
- Líneas siguientes: cada una un segmento $x_1\ y_1\ x_2\ y_2$ en el espacio de **esquinas**, donde $X$ indexa filas (vertical, $0$ arriba) y $Y$ indexa columnas (horizontal, $0$ a la izquierda) — convención idéntica a la imagen del enunciado.

De los 65 segmentos, **37 son muros internos** (separan dos celdas válidas) y los 28 restantes pertenecen al borde exterior, que el chequeo de límites ya cubre.

| Elemento | Valor |
|---|---|
| Celda inicial $s_0$ | $(6,\,0)$ |
| Celda meta $s_T$ | $(1,\,6)$ |
| Distancia Manhattan $s_0 \to s_T$ | $11$ |
| Camino óptimo (BFS sobre el grafo) | $25$ pasos |

El ambiente es **determinista** y **completamente observable**: una misma acción desde el mismo estado produce siempre la misma transición, y el agente conoce su posición exacta. El objetivo es llegar de $s_0$ a $s_T$ minimizando el número de pasos.

Formalmente lo modelamos como un MDP episódico $(\mathcal{S},\, \mathcal{A},\, T,\, r,\, \gamma)$ con estado terminal $s_T$. Las tres componentes que pide la entrega ($\mathcal{S}$, $\mathcal{A}$, $r$) se detallan a continuación.

## 1. Conjunto de estados

El estado se codifica como una tupla $(x, y)$ con la posición del agente en la grilla:

| Componente | Tipo | Rango | Descripción |
|---|---|---|---|
| $x$ | `int` | $\{0, 1, \dots, 7\}$ | Fila de la celda donde está el agente |
| $y$ | `int` | $\{0, 1, \dots, 6\}$ | Columna de la celda donde está el agente |

$$\mathcal{S} = \{(x, y) : 0 \le x < 8,\ 0 \le y < 7\}, \qquad |\mathcal{S}| = 8 \times 7 = 56.$$

- **Estado inicial**: $s_0 = (6, 0)$, fijo en cada episodio.
- **Estado terminal**: $s_T = (1, 6)$. Alcanzar esta celda termina el episodio exitosamente.
- **Propiedad de Markov**: el par $(x, y)$ es información suficiente para decidir, porque la geometría del laberinto y la regla de transición dependen únicamente del estado actual y de la acción ejecutada — no se requiere historial.

La representación es directamente indexable y se usa como clave de la Q-tabla.

## 2. Conjunto de acciones

El agente dispone de **4 acciones discretas** correspondientes a movimientos a 4-conexa en la grilla:

$$\mathcal{A} = \{\text{UP},\ \text{DOWN},\ \text{LEFT},\ \text{RIGHT}\}, \qquad |\mathcal{A}| = 4.$$

Cada acción tiene asociado un desplazamiento $\Delta_a \in \mathbb{Z}^2$:

| Índice | Acción | $\Delta_a = (\Delta x,\, \Delta y)$ |
|:---:|:---:|:---:|
| 0 | UP    | $(-1,\ 0)$ |
| 1 | DOWN  | $(+1,\ 0)$ |
| 2 | LEFT  | $(0,\ -1)$ |
| 3 | RIGHT | $(0,\ +1)$ |

### Aplicabilidad

Las cuatro acciones están **disponibles desde cualquier estado** $s \in \mathcal{S}$. Lo que varía con el estado no es la disponibilidad sino el efecto:

$$
T(s, a) =
\begin{cases}
s + \Delta_a & \text{si } s + \Delta_a \in \mathcal{S} \text{ y no hay muro entre } s \text{ y } s + \Delta_a, \\
s & \text{en caso contrario (intento fuera del grid o muro bloqueante).}
\end{cases}
$$

Cuando la transición devuelve $s$ (la acción no produjo desplazamiento) decimos que ocurrió un *bump*: la acción se "ejecutó" pero el agente permaneció en su celda. Esta convención —acciones siempre aplicables, transiciones inválidas mapeadas a *no-op*— evita máscaras de acción por estado y es estándar en gridworlds tabulares.

## 3. Función de recompensa

La recompensa depende exclusivamente del estado destino de la transición:

$$
r(s, a, s') =
\begin{cases}
+100 & \text{si } s' = s_T, \\
-1   & \text{en cualquier otro caso.}
\end{cases}
$$

### Situaciones explícitas (parejas $(s, a)$)

Toda transición desde un estado no terminal cae en exactamente uno de estos tres casos:

| # | Situación | $s'$ resultante | $r$ |
|:---:|---|---|---:|
| (i)  | $s$ adyacente a $s_T$, $a$ orienta hacia $s_T$ y no hay muro entre ambos | $s_T$ | $+100$ |
| (ii) | $a$ produce un desplazamiento válido a una celda distinta de $s_T$ | $s + \Delta_a$ | $-1$ |
| (iii)| $a$ intenta salir del grid o cruzar un muro (*bump*) | $s$ | $-1$ |

Los casos (ii) y (iii) tienen la misma recompensa pero distinto $s'$; los separamos porque la rúbrica pide enumerar las situaciones de forma explícita.

### Justificación del diseño

- **Costo de paso $-1$.** La única manera de maximizar el retorno es minimizar el número de pasos. Sin él, cualquier política que eventualmente llegue al goal sería óptima; con el costo, la única política óptima es la trayectoria más corta.
- **Recompensa terminal $+100$ frente al costo $-1$.** Asegura que la señal terminal domina aun cuando el episodio sea largo: en el camino óptimo de 25 pasos el retorno es $24\cdot(-1) + 100 = +76$, mientras que el peor caso truncado a 200 pasos da $-200$. La diferencia mantiene gradiente positivo hacia el goal.
- **Bump sin penalización adicional.** El costo de paso ya penaliza el *bump* implícitamente (es un paso desperdiciado). Una penalización extra aceleraría la convergencia pero introduciría un hiperparámetro innecesario para un ambiente de 56 estados, donde el algoritmo ya converge en miles de episodios.
- **Sin *reward shaping* intermedio** (por ejemplo, bonus por acercarse al goal). Mantenemos la formulación canónica de gridworld para que la política óptima coincida exactamente con el camino BFS más corto — eso facilita validar al agente entrenado contra una referencia objetiva.

## Modelo de transición y dinámica del episodio

- **Transición.** Determinista: $T(s' \mid s, a) = 1$ para el $s'$ definido en §2 y $0$ en los demás.
- **Terminación.** El episodio termina ($\texttt{terminated}=\texttt{True}$) al alcanzar $s_T$. Si se exceden $200$ pasos sin llegar, se trunca ($\texttt{truncated}=\texttt{True}$); la truncación **no** corta el *bootstrap* de Q-learning porque no es un estado terminal real.
- **Factor de descuento.** Adoptamos $\gamma = 0.99$ para que la señal terminal mantenga influencia a lo largo de los $25$ pasos del camino óptimo: $\gamma^{24} \approx 0.79$.

## Referencias visuales

- `notebooks/01_exploration.ipynb` carga el laberinto, reproduce la figura del enunciado y calcula el camino óptimo BFS que sirve de referencia de evaluación.
- `notebooks/02_experiments.ipynb` entrena al agente y produce las gráficas de aprendizaje y el GIF del comportamiento aprendido.
