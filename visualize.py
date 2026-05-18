"""
Visualización estilo MiniGrid - Proyecto Final de Aprendizaje por Refuerzo.
Juan Sebastian Paez Cortes - Jean Betancourt Padilla

Genera un GIF animado que replica la apariencia del ambiente original:
  - Fondo gris oscuro (paredes exteriores)
  - Celdas negras (piso navegable)
  - Pared gris divisoria
  - Agente: triángulo rojo apuntando hacia abajo
  - Llave: forma de llave azul
  - Bola: círculo rojo
  - Puerta cerrada: cuadrado azul sólido con símbolo de cerradura
  - Puerta abierta: recuadro azul vacío
  - Meta: rectángulo azul con líneas internas (como ventana)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, FancyArrowPatch, Polygon
from PIL import Image

from environment import (TwoRoomsEnv, ACTION_NAMES, N_ACTIONS,
                         WALL, KEY, BALL, DOOR_CLOSED, DOOR_OPEN, GOAL, EMPTY)
from agent import QLearningAgent


# ============================================================
# Paleta de colores (replicando la imagen original)
# ============================================================
C_OUTER_BG  = "#7A7A7A"   # Gris exterior (zonas de pared/borde)
C_FLOOR     = "#111111"   # Negro (celdas de piso navegable)
C_WALL      = "#8A8A8A"   # Gris (columna divisoria)
C_AGENT     = "#FF3333"   # Rojo (triángulo del agente)
C_KEY       = "#4455FF"   # Azul (llave)
C_BALL      = "#FF3333"   # Rojo (bola)
C_DOOR_C    = "#3344CC"   # Azul sólido (puerta cerrada)
C_DOOR_O    = "#333388"   # Azul oscuro contorno (puerta abierta)
C_GOAL      = "#3344CC"   # Azul (meta / ventana)
C_TEXT      = "#EEEEEE"   # Texto blanco
C_STATUS_BG = "#222222"   # Fondo barra de estado

CELL = 1.0   # tamaño de celda


def _floor_color(env, x, y):
    """Color de fondo de la celda: negro si es piso, gris si es pared."""
    # Columna divisoria (wall col)
    if x == env.WALL_COL and (x, y) != env.DOOR_POS:
        return C_WALL
    # Fuera de límites → exterior gris
    if x < 0 or x >= env.WIDTH or y < 0 or y >= env.HEIGHT:
        return C_OUTER_BG
    return C_FLOOR


def draw_key(ax, cx, cy, color=C_KEY, size=0.28):
    """Dibuja una llave: círculo + varilla + dientes."""
    # Círculo de la llave
    circ = Circle((cx, cy + size * 0.6), size * 0.38,
                  facecolor=color, edgecolor="white", linewidth=1.2, zorder=4)
    ax.add_patch(circ)
    # Hueco del círculo
    hole = Circle((cx, cy + size * 0.6), size * 0.18,
                  facecolor=C_FLOOR, edgecolor="none", zorder=5)
    ax.add_patch(hole)
    # Varilla
    ax.plot([cx, cx], [cy - size * 0.6, cy + size * 0.28],
            color=color, linewidth=3.5, solid_capstyle='round', zorder=4)
    # Dientes
    ax.plot([cx, cx + size * 0.25], [cy - size * 0.15, cy - size * 0.15],
            color=color, linewidth=2.5, solid_capstyle='round', zorder=4)
    ax.plot([cx, cx + size * 0.25], [cy - size * 0.4, cy - size * 0.4],
            color=color, linewidth=2.5, solid_capstyle='round', zorder=4)


def draw_agent(ax, cx, cy, has_key=False, size=0.32):
    """Triángulo rojo apuntando hacia abajo ▼ (como en la imagen original)."""
    tri = Polygon([
        [cx,            cy - size * 0.6],    # punta INFERIOR (vértice hacia abajo)
        [cx - size,     cy + size * 0.5],    # esquina sup izq
        [cx + size,     cy + size * 0.5],    # esquina sup der
    ], closed=True, facecolor=C_AGENT, edgecolor="white", linewidth=1.5, zorder=6)
    ax.add_patch(tri)
    if has_key:
        # Pequeño ícono de llave sobre el agente
        badge = Circle((cx + size * 0.65, cy + size * 0.65), size * 0.2,
                       facecolor=C_KEY, edgecolor="white", linewidth=1, zorder=7)
        ax.add_patch(badge)


def draw_ball(ax, cx, cy, size=0.3):
    """Círculo rojo (bola)."""
    ball = Circle((cx, cy), size,
                  facecolor=C_BALL, edgecolor="white", linewidth=1.5, zorder=4)
    ax.add_patch(ball)


def draw_door_closed(ax, x, y, size=CELL):
    """Puerta cerrada: cuadrado azul sólido con símbolo de cerrojo (−)."""
    r = patches.Rectangle(
        (x * size + 0.02, y * size + 0.02),
        size - 0.04, size - 0.04,
        facecolor=C_DOOR_C, edgecolor="#7799FF", linewidth=1.5, zorder=3
    )
    ax.add_patch(r)
    cx = x * size + size / 2
    cy = y * size + size / 2
    # Símbolo de cerrojo (línea horizontal)
    ax.plot([cx - 0.15, cx + 0.15], [cy, cy],
            color="white", linewidth=2.5, solid_capstyle='round', zorder=4)


def draw_door_open(ax, x, y, size=CELL):
    """Puerta abierta: contorno azul."""
    r = patches.Rectangle(
        (x * size + 0.08, y * size + 0.08),
        size - 0.16, size - 0.16,
        facecolor="none", edgecolor=C_DOOR_O, linewidth=2.5, zorder=3
    )
    ax.add_patch(r)


def draw_goal(ax, x, y, size=CELL):
    """Meta: rectángulo azul con divisores internos (icono de ventana/puerta)."""
    cx = x * size
    cy = y * size
    pad = 0.1
    w = size - 2 * pad
    h = size - 2 * pad

    # Marco externo
    r = patches.Rectangle(
        (cx + pad, cy + pad), w, h,
        facecolor="none", edgecolor=C_GOAL, linewidth=2.5, zorder=4
    )
    ax.add_patch(r)

    # Línea horizontal divisoria
    mid_y = cy + pad + h * 0.45
    ax.plot([cx + pad, cx + pad + w], [mid_y, mid_y],
            color=C_GOAL, linewidth=2, zorder=5)

    # Línea vertical divisoria
    mid_x = cx + pad + w * 0.5
    ax.plot([mid_x, mid_x], [mid_y, cy + pad + h],
            color=C_GOAL, linewidth=2, zorder=5)


def draw_frame(env, step_num, action_name=None, reward=None,
               event=None, total_reward=0.0, figsize=(13, 7)):
    """Dibuja un frame del ambiente con estética MiniGrid."""
    W, H = env.WIDTH, env.HEIGHT

    # Tamaño total de la figura en celdas: añadimos borde exterior de 0.5
    BORDER = 0.6
    fig_w = W + 2 * BORDER
    fig_h = H + 2 * BORDER

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(C_OUTER_BG)

    # Fondo gris externo completo
    bg = patches.Rectangle(
        (-BORDER, -BORDER), fig_w, fig_h,
        facecolor=C_OUTER_BG, edgecolor="none", zorder=0
    )
    ax.add_patch(bg)

    # Dibujar celdas de piso
    for y in range(H):
        for x in range(W):
            fc = _floor_color(env, x, y)
            r = patches.Rectangle(
                (x * CELL, (H - 1 - y) * CELL), CELL, CELL,
                facecolor=fc, edgecolor="#2A2A2A", linewidth=0.4, zorder=1
            )
            ax.add_patch(r)

    # Dibujar objetos
    for y in range(H):
        for x in range(W):
            cx = x * CELL + CELL / 2
            cy = (H - 1 - y) * CELL + CELL / 2
            content = env._cell_content(x, y)

            if content == KEY:
                draw_key(ax, cx, cy)

            elif content == BALL:
                draw_ball(ax, cx, cy)

            elif content == DOOR_CLOSED:
                draw_door_closed(ax, x, H - 1 - y)

            elif content == DOOR_OPEN:
                draw_door_open(ax, x, H - 1 - y)

            elif content == GOAL:
                draw_goal(ax, x, H - 1 - y)

    # Dibujar agente
    ax_pos, ay_pos = env.agent_pos
    acx = ax_pos * CELL + CELL / 2
    acy = (H - 1 - ay_pos) * CELL + CELL / 2
    draw_agent(ax, acx, acy, has_key=bool(env.has_key))

    # Configurar ejes
    ax.set_xlim(-BORDER, W * CELL + BORDER)
    ax.set_ylim(-2.0, H * CELL + BORDER)
    ax.set_aspect('equal')
    ax.axis('off')

    # Título
    ax.text(W * CELL / 2, H * CELL + BORDER * 0.6,
            "Agente Q-learning — Ambiente de Dos Habitaciones",
            fontsize=13, ha='center', va='center',
            color=C_TEXT, fontweight='bold')

    # Barra de estado
    if action_name:
        line1 = f"Paso {step_num:2d}  |  Acción: {action_name}"
        if reward is not None:
            line1 += f"  |  R = {reward:+.1f}"
        if event:
            line1 += f"  |  {event}"
        line1 += f"  |  R_total = {total_reward:+.1f}"
    else:
        line1 = "Estado Inicial"

    state = env._get_state()
    line2 = (f"S = (x={state[0]}, y={state[1]}, llave={state[2]}, "
             f"puerta={state[3]}, bola_removida={state[4]})")

    ax.text(W * CELL / 2, -0.55, line1,
            fontsize=10.5, ha='center', va='center',
            color="#DDDDDD", fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=C_STATUS_BG,
                      edgecolor='#555555', alpha=0.95))

    ax.text(W * CELL / 2, -1.35, line2,
            fontsize=9, ha='center', va='center',
            color="#999999", fontfamily='monospace')

    plt.tight_layout()
    return fig


def generate_demo(qtable_path="qtable.pkl", output="demo_agente.gif",
                  save_frames=False, fps=2):
    env = TwoRoomsEnv()
    agent = QLearningAgent()
    agent.load_qtable(qtable_path)
    agent.epsilon = 0.0

    state = env.reset()
    frames = []

    if save_frames:
        os.makedirs("frames", exist_ok=True)

    def save_fig(fig, idx):
        fig.savefig("_tmp.png", dpi=110, facecolor=fig.get_facecolor(),
                    bbox_inches='tight', pad_inches=0.15)
        plt.close(fig)
        img = Image.open("_tmp.png").copy()
        frames.append(img)
        if save_frames:
            img.save(f"frames/frame_{idx:02d}.png")

    # Frame inicial
    fig = draw_frame(env, 0)
    save_fig(fig, 0)

    done = False
    total_r = 0.0
    step = 0

    while not done:
        a = agent.select_action(state, greedy=True)
        state, r, done, info = env.step(a)
        total_r += r
        step += 1
        fig = draw_frame(env, step, ACTION_NAMES[a], r, info.get("event"), total_r)
        save_fig(fig, step)

    # Pausa al final
    for _ in range(4):
        frames.append(frames[-1].copy())

    if os.path.exists("_tmp.png"):
        os.remove("_tmp.png")

    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=False
    )

    print(f"\n[DEMO] GIF guardado en '{output}' ({len(frames)} frames, {fps} fps)")
    print(f"[DEMO] Pasos: {step} | Recompensa total: {total_r:.2f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--qtable", type=str, default="qtable.pkl")
    p.add_argument("--output", type=str, default="demo_agente.gif")
    p.add_argument("--save-frames", action="store_true")
    p.add_argument("--fps", type=int, default=2)
    args = p.parse_args()
    generate_demo(args.qtable, args.output, args.save_frames, args.fps)
