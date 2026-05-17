"""Curvas de aprendizaje del entrenamiento."""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def moving_avg(data, w=100):
    data = np.asarray(data, dtype=float)
    return np.convolve(data, np.ones(w)/w, mode="valid") if len(data) >= w else data


def plot(log_path="training_log.json", output="learning_curves.png"):
    with open(log_path) as f:
        log = json.load(f)
    rewards, steps, success = log["rewards"], log["steps"], log["success"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Curvas de Aprendizaje — Q-learning en Ambiente de Dos Habitaciones",
                 fontsize=14, fontweight='bold', y=1.01)

    w = 100
    ma_r = moving_avg(rewards, w)
    axes[0].plot(rewards, alpha=0.2, color="steelblue", linewidth=0.5)
    axes[0].plot(np.arange(len(ma_r)) + w - 1, ma_r, color="navy",
                 linewidth=2, label=f"Media móvil ({w})")
    axes[0].set_title("Recompensa Acumulada por Episodio")
    axes[0].set_xlabel("Episodio"); axes[0].set_ylabel("Recompensa")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    ma_s = moving_avg(steps, w)
    axes[1].plot(steps, alpha=0.2, color="darkorange", linewidth=0.5)
    axes[1].plot(np.arange(len(ma_s)) + w - 1, ma_s, color="chocolate",
                 linewidth=2, label=f"Media móvil ({w})")
    axes[1].set_title("Pasos por Episodio")
    axes[1].set_xlabel("Episodio"); axes[1].set_ylabel("Pasos")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    w2 = 200
    ma_ok = moving_avg(success, w2)
    axes[2].plot(np.arange(len(ma_ok)) + w2 - 1, ma_ok * 100,
                 color="seagreen", linewidth=2)
    axes[2].set_title(f"Tasa de Éxito (ventana {w2})")
    axes[2].set_xlabel("Episodio"); axes[2].set_ylabel("% Éxito")
    axes[2].set_ylim(-5, 105); axes[2].grid(alpha=0.3)
    axes[2].axhline(100, color='green', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"[PLOT] Gráficas guardadas en '{output}'")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=str, default="training_log.json")
    p.add_argument("--output", type=str, default="learning_curves.png")
    args = p.parse_args()
    plot(args.log, args.output)
