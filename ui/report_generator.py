import os
import json


def generate_report(log_path="training_log.json", qtable_path="qtable.pkl",
                    output="reporte.md", agent_params=None):
    log = None
    if os.path.exists(log_path):
        with open(log_path) as f:
            log = json.load(f)

    success_ep = None
    if log and log.get("success"):
        for i, s in enumerate(log["success"], 1):
            if s == 1:
                success_ep = i
                break

    eval_text = ""
    try:
        from evaluate import evaluate
        all_r, all_s, all_ok = evaluate(qtable_path, n_episodes=10, render=False)
        import numpy as np
        eval_text = (
            f"| Éxito | {np.mean(all_ok)*100:.1f}% |\n"
            f"| Recompensa prom | {np.mean(all_r):.2f} ± {np.std(all_r):.2f} |\n"
            f"| Pasos prom | {np.mean(all_s):.1f} |\n"
        )
    except Exception:
        eval_text = "| (No disponible) | |\n"

    params = agent_params or {}
    alpha = params.get("alpha", 0.1)
    gamma = params.get("gamma", 0.99)
    epsilon = params.get("epsilon", 1.0)
    eps_min = params.get("epsilon_min", 0.05)
    eps_decay = params.get("epsilon_decay", 0.9995)

    n_episodes = log["episodes"] if log else "N/A"
    qtable_states = "N/A"
    if os.path.exists(qtable_path):
        try:
            from agent import QLearningAgent
            a = QLearningAgent()
            a.load_qtable(qtable_path)
            qtable_states = str(len(a.Q))
        except Exception:
            pass

    md = f"""# Proyecto Final — Aprendizaje por Refuerzo
## Definición del Ambiente, Estados y Recompensas

**Materia:** Aprendizaje por Refuerzo  
**Fecha:** Mayo 2026  
**Autores:** Juan Sebastian Paez Cortes — Jean Betancourt Padilla

---

## 1. Ambiente

Grid de dos habitaciones (9 columnas × 5 filas) separadas por una pared
vertical en la columna x=4, con una puerta en la posición (4, 3).

El agente debe completar la siguiente secuencia de sub-tareas:
1. Remover la bola que bloquea la puerta → acción *PICKUP* en (2, 3)
2. Recoger la llave → acción *PICKUP* en (3, 1)
3. Abrir la puerta → acción *TOGGLE* frente a (4, 3)
4. Navegar hasta la meta → posición (6, 0)

**Acciones disponibles (6):** `UP`, `DOWN`, `LEFT`, `RIGHT`, `PICKUP`, `TOGGLE`

**Estado:** Tupla `(x, y, has_key, door_open, ball_removed)` — 360 estados teóricos.

---

## 2. Sistema de recompensas

| Evento | Recompensa |
|--------|-----------|
| Meta alcanzada | **+100** |
| Puerta abierta (1ª vez) | **+20** |
| Llave recogida (1ª vez) | **+10** |
| Bola removida (1ª vez) | **+10** |
| Acción inválida (pared, bad PICKUP, bad TOGGLE) | **−1.0** |
| Costo de vida por paso | **−0.1** |
| Timeout (≥200 pasos) | **−10** |

---

## 3. Agente — Q-learning tabular

| Hiperparámetro | Valor |
|---------------|-------|
| Alpha (α) | {alpha} |
| Gamma (γ) | {gamma} |
| Epsilon inicial (ε) | {epsilon} |
| Epsilon mínimo | {eps_min} |
| Epsilon decay | {eps_decay} |

La Q-table es un `defaultdict` indexado por la tupla de estado, con 6
Q-valores por acción. Se usa selección de acción ε-greedy con decaimiento
exponencial por episodio.

---

## 4. Resultados del entrenamiento

| Métrica | Valor |
|---------|-------|
| Episodios ejecutados | {n_episodes} |
| Meta alcanzada en episodio | {success_ep if success_ep else "No alcanzada"} |
| Estados en Q-table | {qtable_states} |

---

## 5. Evaluación (modo greedy, ε=0)

{eval_text}

---

## 6. Archivos generados

| Archivo | Descripción |
|---------|------------|
| `qtable.pkl` | Q-table entrenada (pickle) |
| `qtable.txt` | Q-table exportada en texto legible |
| `training_log.json` | Historial de recompensas, pasos, éxito por episodio |
| `learning_curves.png` | Curvas de aprendizaje (3 gráficas) |
| `demo_agente.gif` | Demo del agente resolviendo (estilo MiniGrid) |
| `training_video.mp4` | Video del episodio ganador |
| `reporte.md` | Este reporte |

![Curvas de aprendizaje](learning_curves.png)
"""
    with open(output, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[REPORT] Reporte generado en '{output}'")
