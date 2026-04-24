# Cambios — de la versión inicial a la versión que resuelve el entorno

Documento del trayecto desde `experiments.ipynb` original (versión que **no aprendía** —
tasa de éxito 0/100, recompensa promedio −5.76) hasta la versión actual
(100/100 de éxito, recompensa promedio ≈ +2.77).

## Diagnóstico inicial

Con la versión inicial el agente:
- Entrenaba 2000 episodios con recompensa promedio estancada alrededor de −5 (≈ costo por paso acumulado).
- En evaluación greedy no completaba ninguna subtarea: 0/100 de éxito, recompensa −5.76 ≈ 576 pasos × −0.01.
- El GIF mostraba al agente dando vueltas sin avanzar.

Cinco causas identificadas:

1. **Estado incompleto**: `get_state()` marcaba `has_key` como sticky, pero no distinguía qué objeto estaba cargando el agente *ahora*. Con la bola en la mano, intentar `pickup` de la llave fallaba, pero el estado parecía idéntico al de "ya solté la bola". Política ambigua ⇒ no aprende.
2. **Costo por paso dominante**: `-0.01 × 576 = −5.76`, casi igual en magnitud a la suma de bonus de subtarea (+1.0). El gradiente de recompensa no favorecía completar subtareas.
3. **Bootstrap incorrecto en truncación**: el update de Q-learning usaba `done = terminated or truncated`, poniendo `best_next = 0` al truncar. Pero truncar por `max_steps` **no** es un estado terminal: debe hacer bootstrap normal.
4. **Política greedy con ciclos**: `_argmax` devolvía el primer índice con empate. Los Q-valores igualados producían ciclos deterministas; durante entrenamiento ε-greedy los rompía con exploración, pero la evaluación greedy se atascaba.
5. **Shaping incompleto**: sin bonus para las subtareas finales (soltar la llave, recoger la caja), el agente llegaba hasta "puerta abierta" y no encontraba señal para el último tramo.

## Cambios aplicados

### `src/env.py`

- **Estado**: reemplazado `has_key` (bool sticky) por `carrying_type` (string `"none" | "key" | "ball" | "box"`). El estado ahora es `(pos, direction, carrying_type, ball_moved, door_open)`.
- **Costo por paso**: bajado de `-0.01` a `-0.001` para no dominar los bonus.
- **Subtareas adicionales**: `key_dropped` (soltar la llave después de abrir la puerta, +0.3) y `box_picked` (recoger la caja, +0.5). Guían al agente por el tramo final.

### `src/agent.py`

- **`_argmax` con desempate aleatorio**: entre índices con Q-valor máximo, elige uno al azar. Rompe ciclos en políticas greedy deterministas.

### `notebooks/experiments.ipynb`

- **Hiperparámetros**:
  - `N_EPISODES`: 2 000 → 10 000.
  - `MAX_STEPS`: 300 → 576 (coincide con el límite del env MiniGrid).
  - `epsilon_decay`: 0.995 → 0.9995 (decaimiento más lento, más exploración efectiva).
  - `N_EVAL`: 20 → 100.
- **Reproducibilidad**: `random.seed(42)` al inicio del entrenamiento.
- **Fix de bootstrap**: el update ahora recibe `terminated` en lugar de `done`.
- **Artefacto**: la Q-tabla se guarda como `qtable_v2.pkl` (la `v1` queda como referencia histórica).
- **Visualización**: celda nueva que graba un episodio greedy como GIF en `video/episode_greedy.gif`.

### `train.py` (nuevo)

Script headless equivalente al notebook. Útil para iterar sin Jupyter y para CI/reproducibilidad. Reporta también el contador acumulado de episodios exitosos durante el entrenamiento.

## Resultados

| Versión      | Entrenamiento (avg last 500) | Eval greedy (éxito) | Eval greedy (recompensa) |
|--------------|------------------------------|---------------------|--------------------------|
| Inicial      | ≈ −5.0                       | 0/100               | −5.76                    |
| Intermedia   | ≈ +1.9                       | 0/100               | +0.42                    |
| Final        | ≈ +2.57                      | 100/100             | +2.78                    |

La versión intermedia (después de arreglar estado, costo, bootstrap y tie-breaking) ya aprendía durante el entrenamiento, pero la Q-tabla resultante dependía del RNG: algunos runs producían greedy que se atascaba después de abrir la puerta. Los bonus de `key_dropped` y `box_picked` más la semilla fija estabilizan el resultado.
