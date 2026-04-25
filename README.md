# Proyecto RL — Laberinto

Agente de Q-learning tabular para resolver un laberinto rectangular de 8×7 celdas con muros internos, definido por el archivo del enunciado (`data/project_lab_v2.txt`). El agente parte de la celda `(6, 0)` y debe llegar a la meta `(1, 6)` minimizando el número de pasos.

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

El flujo del proyecto vive en los notebooks de `notebooks/`:

- **`01_exploration.ipynb`** — carga el laberinto, lo visualiza, calcula el camino óptimo con BFS (referencia de evaluación) y prueba la API del ambiente con acciones manuales.
- **`02_experiments.ipynb`** — entrena el agente Q-learning, guarda la Q-tabla, genera la curva de aprendizaje, evalúa la política greedy y graba un GIF del episodio.

El código reutilizable vive en `src/`:

- `src/maze.py` — parser del archivo del laberinto, geometría (`Maze`) y ambiente Q-learning (`MazeEnv`).
- `src/agent.py` — implementación del agente Q-learning tabular con persistencia.

## Estructura

```
Proyecto/
├── data/
│   └── project_lab_v2.txt       # archivo del enunciado (definición del laberinto)
├── src/
│   ├── maze.py
│   └── agent.py
├── notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_experiments.ipynb
├── docs/
│   └── partial.md               # entrega parcial: caracterización (estados, acciones, recompensa)
├── results/                     # Q-tabla, curvas, gráficas
├── video/                       # GIF de la demo greedy
└── _archive/                    # iteraciones anteriores del proyecto (referencia histórica)
```

## Entregas

- **Entrega parcial (semana 5)**: `docs/partial.md`. Caracterización completa del problema según los tres requisitos del enunciado (definición del conjunto de estados, definición de acciones con su aplicabilidad, función de recompensa con valores numéricos por situación).
