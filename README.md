# Proyecto RL — DoorKey

Agente de Q-learning tabular para resolver un ambiente tipo MiniGrid con dos habitaciones separadas por una puerta con llave. El agente debe retirar una bola que bloquea la puerta, recoger la llave, abrir la puerta y llegar a la casilla objetivo.

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

El entrenamiento y la evaluación se ejecutan desde los notebooks en `notebooks/`:

- `exploration.ipynb` — exploración del ambiente y pruebas rápidas.
- `experiments.ipynb` — entrenamiento, evaluación y análisis.

El código reutilizable (ambiente y agente) vive en `src/` y se importa desde los notebooks.

## Estructura

- `src/` — ambiente y agente (código reutilizable).
- `notebooks/` — experimentación.
- `docs/` — entregas parcial y final.
- `results/` — Q-tablas, logs y gráficas.
- `video/` — demo final.
