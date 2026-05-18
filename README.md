# Two Rooms RL — Proyecto Final de Aprendizaje por Refuerzo

## Datos del proyecto

| Campo | Valor |
|-------|-------|
| **Título** | Definición del Ambiente, Estados y Recompensas |
| **Materia** | Aprendizaje por Refuerzo |
| **Fecha** | Mayo 2026 |
| **Autores** | Juan Sebastian Paez Cortes — Jean Betancourt Padilla |

## Descripción

Ambiente de dos habitaciones separadas por una pared con puerta y llave. El agente
debe remover una bola que bloquea la puerta, recoger la llave, abrir la puerta y
navegar hasta la meta en el cuarto derecho. Se implementa un agente Q-learning
tabular con interfaz gráfica en PyQt6 para entrenamiento, visualización y análisis.

---

## Arquitectura del proyecto

### Archivos del ambiente y agente (NO modificar salvo `gym_env.py`)

| Archivo | Descripción |
|---------|-------------|
| `environment.py` | `TwoRoomsEnv` — grid determinista 9×5 con 6 acciones (UP, DOWN, LEFT, RIGHT, PICKUP, TOGGLE). Estado: tupla `(x, y, has_key, door_open, ball_removed)`. Sistema de recompensas: +100 goal, +20 puerta, +10 llave, +10 bola, −1 inválido, −0.1 living, −10 timeout. |
| `agent.py` | `QLearningAgent` — agente tabular con Q-table (`defaultdict`), selección ε-greedy, decaimiento exponencial de epsilon. Métodos: `select_action()`, `update()`, `decay_epsilon()`, `save_qtable()`, `load_qtable()`, `export_qtable_readable()`. |
| `gym_env.py` | `TwoRoomsGymEnv(gymnasium.Env)` — wrapper que adapta `TwoRoomsEnv` a la API estándar de Gymnasium (`reset`, `step`, `observation_space`, `action_space`) sin modificar los originales. |

### Archivos de entrenamiento y evaluación (NO modificar)

| Archivo | Descripción |
|---------|-------------|
| `train.py` | Entrenamiento standalone por CLI. Función `train(episodes, save_path, log_path)`. Corre episodios Q-learning, guarda `qtable.pkl` y `training_log.json`. Hiperparámetros fijos. |
| `evaluate.py` | Evaluación en modo greedy (ε=0). Función `evaluate(qtable_path, n_episodes, render)`. Retorna `(all_r, all_s, all_ok)` con métricas por episodio. |
| `plot_results.py` | Genera `learning_curves.png` con 3 subplots desde `training_log.json`: recompensa acumulada, pasos por episodio, tasa de éxito (con medias móviles). Función `plot(log_path, output)`. |
| `visualize.py` | Genera `demo_agente.gif` de alta calidad (estilo MiniGrid) usando matplotlib + PIL. Dibuja grid, objetos, agente (triángulo rojo), llave, bola, puerta, goal y barra de estado. Función `generate_demo(qtable_path, output, save_frames, fps)`. |

### Interfaz gráfica (UI con PyQt6)

| Archivo | Descripción |
|---------|-------------|
| `main.py` | Entry point. Crea `QApplication` e instancia `MainWindow`. |
| `ui/__init__.py` | Inicializador del paquete `ui`. |
| `ui/main_window.py` | `MainWindow` — QMainWindow con `QTabWidget` de 2 pestañas (Entrenamiento, Q-Table). Tema oscuro Dracula-like. Encabezado con título, autores, materia y fecha. |
| `ui/training_tab.py` | `TrainingTab` — Panel izquierdo: hiperparámetros (α, γ, ε, ε_min, ε_decay, episodios, velocidad), checkbox "Detener al alcanzar meta", botones Iniciar/Pausar/Detener, barra de progreso. Panel derecho: `GridWidget` (visualización en vivo), `MetricsWidget` (gráficas pyqtgraph), `VideoPlayer` (video), visor de GIF/PNG. Grupo post-entrenamiento: Evaluar agente, Curvas de aprendizaje, Generar demo GIF, Generar reporte. |
| `ui/training_thread.py` | `TrainingThread(QThread)` — Ejecuta episodios en hilo separado. Soporta pausa/reanudar/detener mediante mutex. Emite señales: `step_signal`, `episode_signal`, `progress_signal`, `goal_reached_signal`, `finished_signal`. Guarda `qtable.pkl` y `training_log.json` al finalizar. Parámetro `stop_on_goal` para controlar parada temprana. |
| `ui/grid_widget.py` | `GridWidget` — Renderiza el grid 9×5 con `QPainter`. Fondo blanco, solo colores en agente (azul), llave (dorado), bola (rojo), puerta (marrón/verde) y goal (verde). Paredes con líneas diagonales. Muestra paso, acción, recompensa y evento actual. |
| `ui/metrics_widget.py` | `MetricsWidget` — 3 gráficas pyqtgraph en tiempo real: recompensa acumulada por episodio, epsilon decay, pasos por episodio. Método `add_episode()` para actualizar. |
| `ui/qtable_tab.py` | `QTableTab` — Visor tabular de Q-values (12 columnas: estado + 6 acciones). Botones: Exportar .txt, Cargar Q-table (.pkl), Refrescar desde agente. |
| `ui/video_player.py` | `VideoPlayer` — Reproductor basado en OpenCV (`cv2.VideoCapture` + `QTimer`). Muestra `training_video.mp4` frame por frame en un `QLabel`. Botones: Reproducir/Pausar, Descargar video. |
| `ui/report_generator.py` | `generate_report()` — Genera `reporte.md` automático con ambiente, sistema de recompensas, hiperparámetros, resultados del entrenamiento, evaluación y lista de archivos generados. |

### Archivos de configuración

| Archivo | Descripción |
|---------|-------------|
| `.gitignore` | Excluye `/venv`, `/examples`, `/proyecto`, `__pycache__/`, `agents.md`, `*.mp4`, `*.gif`, `*.png`, `reporte.md`, `*.pkl`, `*.txt`, `*.json`, `*.toml`. |

### Archivos generados en runtime (gitignored)

| Archivo | Origen | Descripción |
|---------|--------|-------------|
| `qtable.pkl` | `TrainingThread` / `train.py` | Q-table entrenada (pickle binario) |
| `qtable.txt` | `QTableTab` | Q-table exportada en texto legible |
| `training_log.json` | `TrainingThread` / `train.py` | Historial de recompensas, pasos, éxito |
| `learning_curves.png` | `plot_results.py` | Curvas de aprendizaje (3 gráficas) |
| `demo_agente.gif` | `visualize.py` | Demo del agente resolviendo (MiniGrid) |
| `training_video.mp4` | `TrainingTab` | Video del episodio ganador (OpenCV) |
| `reporte.md` | `ui/report_generator.py` | Reporte automático en Markdown |

---

## Dependencias

```
Python >= 3.12
numpy >= 2.0
PyQt6 >= 6.7
pyqtgraph >= 0.13
opencv-python-headless >= 4.10
gymnasium >= 1.0
matplotlib >= 3.8
pillow >= 10.0
```

### Instalación con `uv`

```bash
uv pip install pyqt6 pyqtgraph opencv-python-headless gymnasium matplotlib pillow --python venv/bin/python3
```

---

## Ejecución

```bash
python main.py
```

---

## Flujo de uso

1. Abrir la aplicación con `python main.py`.
2. Ajustar hiperparámetros (α, γ, ε, etc.) en la pestaña **Entrenamiento**.
3. Marcar/desmarcar **"Detener al alcanzar la meta"** según prefieras parada temprana o correr todos los episodios.
4. Presionar **Iniciar entrenamiento**.
5. Observar al agente en vivo en el grid y las métricas actualizarse.
6. Pausar/reanudar/detener según se necesite.
7. Al finalizar (por goal o por episodios), se habilitan las acciones post-entrenamiento:
   - **Evaluar agente**: muestra tasa de éxito, recompensa media, pasos medios.
   - **Curvas de aprendizaje**: genera y muestra PNG con 3 gráficas.
   - **Generar demo GIF**: crea un GIF de alta calidad del agente resolviendo.
   - **Generar reporte**: produce `reporte.md` con todos los resultados.
8. En la pestaña **Q-Table** se puede inspeccionar la tabla, exportar en `.txt` o cargar una Q-table existente (`.pkl`).
9. El video `training_video.mp4` se genera automáticamente según el modo de parada.
