# Entrega final

## 1. Descripción del problema

El ambiente consiste en dos habitaciones de una grilla MiniGrid separadas por una puerta con llave (`MiniGrid-BlockedUnlockPickup-v0`). El agente parte en la habitación izquierda y debe alcanzar una caja objetivo en la habitación derecha. Para lograrlo debe completar una secuencia de subtareas:

1. Recoger la bola que bloquea la puerta y soltarla en otro lugar.
2. Recoger la llave del color correspondiente.
3. Abrir la puerta con la llave.
4. Soltar la llave para liberar el inventario.
5. Navegar hasta la caja objetivo y recogerla.

El agente solo puede cargar un objeto a la vez, lo que introduce dependencias de orden entre las subtareas.

## 2. Representación del estado y acciones

### Estado

El estado se codifica como una tupla hashable para indexar la Q-tabla:

| Componente      | Tipo          | Rango / Valores                         | Descripción                              |
|-----------------|---------------|------------------------------------------|------------------------------------------|
| `pos`           | `(int, int)`  | `(0..W-1, 0..H-1)`                      | Posición del agente en la grilla         |
| `direction`     | `int`         | `0..3`                                   | Orientación (0=E, 1=S, 2=W, 3=N)        |
| `carrying_type` | `str`         | `"none"`, `"key"`, `"ball"`, `"box"`     | Objeto que el agente lleva actualmente   |
| `ball_moved`    | `bool`        | `{False, True}`                          | La bola ya no bloquea la puerta          |
| `door_open`     | `bool`        | `{False, True}`                          | La puerta está abierta                   |

La Q-tabla resultante del entrenamiento contiene **635 estados visitados**.

### Acciones

Se utiliza el conjunto completo de 6 acciones discretas de MiniGrid:

| Índice | Nombre     | Descripción                                       |
|--------|------------|---------------------------------------------------|
| 0      | `left`     | Girar a la izquierda                              |
| 1      | `right`    | Girar a la derecha                                |
| 2      | `forward`  | Avanzar una casilla                               |
| 3      | `pickup`   | Recoger el objeto frente al agente                |
| 4      | `drop`     | Soltar el objeto que lleva                        |
| 5      | `toggle`   | Abrir/cerrar puerta (requiere llave del color)    |

## 3. Función de recompensa (reward shaping)

Se aplica un shaping de recompensa basado en subtareas, con bonificaciones que se entregan una sola vez por episodio:

| Evento                                        | Recompensa |
|------------------------------------------------|-----------:|
| Cada paso (costo de tiempo)                    |    −0.001  |
| Mover la bola por primera vez                  |    +0.20   |
| Recoger la llave por primera vez               |    +0.30   |
| Abrir la puerta por primera vez                |    +0.50   |
| Soltar la llave después de abrir la puerta     |    +0.30   |
| Recoger la caja objetivo                       |    +0.50   |
| Llegar al objetivo (episodio exitoso)          |    +1.00   |

La suma máxima de bonificaciones por episodio es **+2.80**. El costo por paso se fijó en −0.001 (y no −0.01) para evitar que domine los bonus de subtarea.

## 4. Algoritmo: Q-learning tabular

Se emplea Q-learning off-policy con política ε-greedy para la exploración. La regla de actualización es:

```
Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a') − Q(s, a)]
```

donde el bootstrap (`max_a' Q(s', a')`) se anula **solo** cuando el episodio termina por `terminated`, no por `truncated` (alcanzar `max_steps`). Esta distinción es crítica: truncar por tiempo no es un estado terminal y debe hacer bootstrap normal.

### Hiperparámetros

| Parámetro       | Valor   | Justificación                                                  |
|-----------------|---------|----------------------------------------------------------------|
| `alpha`         | 0.1     | Tasa de aprendizaje estándar para entornos deterministas       |
| `gamma`         | 0.99    | Descuento alto: las subtareas están a muchos pasos del final   |
| `epsilon`       | 1.0 → 0.05 | Exploración completa al inicio, mínimo residual de 5%       |
| `epsilon_decay` | 0.9995  | Decaimiento lento: ε alcanza ~0.05 alrededor del episodio 5000 |
| `N_EPISODES`    | 10,000  | Suficiente para convergencia estable                           |
| `MAX_STEPS`     | 576     | Coincide con el límite interno de MiniGrid                     |

### Desempate aleatorio en argmax

La función `_argmax` del agente selecciona aleatoriamente entre las acciones con Q-valor máximo empatado. Esto evita ciclos deterministas en la política greedy, donde empates sistemáticos pueden producir un agente que repite la misma secuencia de acciones indefinidamente.

## 5. Resultados de entrenamiento

### Progresión del entrenamiento (media móvil últimos 500 episodios)

| Episodio | ε     | Recompensa promedio |
|----------|-------|---------------------|
| 500      | 0.779 | −0.015              |
| 1,000    | 0.606 | −0.090              |
| 1,500    | 0.472 | −0.093              |
| 2,000    | 0.368 | +0.131              |
| 2,500    | 0.286 | +0.333              |
| 3,000    | 0.223 | +0.490              |
| 3,500    | 0.174 | +1.269              |
| 4,000    | 0.135 | +2.461              |
| 5,000    | 0.082 | +2.311              |
| 7,500    | 0.050 | +2.748              |
| 10,000   | 0.050 | +2.568              |

Se observan tres fases claras:

1. **Exploración (ep 1–2,000)**: ε alto, el agente explora el espacio de estados sin lograr recompensa significativa. La recompensa promedio ronda −0.09 (dominada por el costo por paso).
2. **Aprendizaje rápido (ep 2,000–4,000)**: ε desciende lo suficiente para que la política aprendida se ejecute con frecuencia. La recompensa salta de +0.13 a +2.46, indicando que el agente domina la secuencia completa de subtareas.
3. **Convergencia (ep 4,000–10,000)**: ε estabilizado en 0.05. La recompensa oscila alrededor de +2.6, cercana al máximo teórico de +2.80. Las fluctuaciones se deben a la exploración residual del 5%.

### Evaluación greedy (100 episodios)

| Métrica               | Valor      |
|------------------------|-----------|
| **Tasa de éxito**      | **100/100** |
| **Recompensa promedio** | **+2.776** |
| **Pasos para resolver** | **24**     |

El agente greedy resuelve el ambiente en 24 pasos de forma consistente, completando todas las subtareas en orden óptimo. La recompensa de +2.776 está muy cerca del máximo teórico de +2.80, con la diferencia explicada por el costo acumulado de los 24 pasos (24 × −0.001 = −0.024).

## 6. Análisis del proceso de desarrollo

### Versión inicial (no aprendía)

La primera implementación presentaba una tasa de éxito de **0/100** y recompensa promedio de **−5.76**. Se identificaron cinco problemas:

| Problema | Causa | Solución aplicada |
|----------|-------|--------------------|
| Estado ambiguo | `has_key` era un flag sticky booleano; no distinguía qué objeto cargaba el agente | Reemplazado por `carrying_type` (string con el tipo de objeto actual) |
| Costo por paso dominante | −0.01 × 576 = −5.76, magnitud comparable a los bonus | Reducido a −0.001 |
| Bootstrap incorrecto | `done = terminated or truncated` anulaba el bootstrap al truncar por tiempo | `done` usa solo `terminated` |
| Ciclos greedy | `argmax` devolvía siempre el primer índice en empates | Desempate aleatorio entre índices máximos |
| Shaping incompleto | Sin señal para el tramo final (soltar llave, recoger caja) | Agregados bonus `key_dropped` (+0.3) y `box_picked` (+0.5) |

### Evolución de versiones

| Versión      | Avg reward (últimos 500) | Éxito greedy | Recompensa greedy |
|--------------|--------------------------|--------------|-------------------|
| Inicial      | ≈ −5.0                   | 0/100        | −5.76             |
| Intermedia   | ≈ +1.9                   | 0/100        | +0.42             |
| **Final**    | **≈ +2.57**              | **100/100**  | **+2.78**         |

La versión intermedia (con fixes de estado, costo, bootstrap y tie-breaking) ya aprendía durante el entrenamiento, pero la evaluación greedy era inestable. Los bonus adicionales para las subtareas finales estabilizaron la política.

## 7. Pruebas realizadas

### Reproducibilidad

- Semilla fija para el ambiente (`seed=0`) y para el agente (`random.seed(42)`).
- Entrenamiento reproducible tanto desde el notebook (`experiments_V2.ipynb`) como desde el script headless (`train.py`).
- Q-tabla serializada en `results/qtable_v2.pkl` para evaluación independiente.

### Evaluación cuantitativa

- **100 episodios greedy**: 100% de éxito, recompensa promedio +2.776.
- **Comparación con versión inicial**: mejora de −5.76 a +2.776 en recompensa promedio.
- **Verificación del bootstrap**: confirmado que usar solo `terminated` (y no `truncated`) produce convergencia correcta.

### Evaluación cualitativa

- **GIF animado** (`video/episode_greedy.gif`): muestra visualmente al agente resolviendo el ambiente en 24 pasos, completando cada subtarea en secuencia.
- **Curva de aprendizaje**: gráfica generada en el notebook mostrando la transición de exploración a explotación.

## 8. Conclusiones

- **Q-learning tabular es suficiente** para este ambiente dado un estado bien diseñado (635 estados únicos) y reward shaping adecuado.
- **La representación del estado es crítica**: la diferencia entre `has_key` (booleano sticky) y `carrying_type` (qué lleva el agente ahora) fue la causa principal del fracaso inicial.
- **El reward shaping guía sin distorsionar**: los bonus de subtarea son señales intermedias que no alteran la política óptima, solo aceleran su descubrimiento.
- **Detalles de implementación importan**: el tratamiento correcto de `terminated` vs. `truncated` y el desempate aleatorio en `argmax` fueron necesarios para la convergencia.
