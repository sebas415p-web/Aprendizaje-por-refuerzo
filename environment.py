"""
Ambiente de dos habitaciones - Proyecto Final de Aprendizaje por Refuerzo.
Juan Sebastian Paez Cortes - Jean Betancourt Padilla

Layout (9 columnas x 5 filas, pared divisoria en columna 4):

    .  .  .  .  #  .  G  .  .     y=0
    .  .  .  K  #  .  .  .  .     y=1
    A  .  .  .  #  .  .  .  .     y=2
    .  .  B  .  D  .  .  .  .     y=3
    .  .  .  .  #  .  .  .  .     y=4

    A=agente  K=llave  B=bola  D=puerta  G=meta  #=pared

Estado: S = (x, y, tiene_llave, puerta_abierta, bola_removida)

Acciones (6):
    0 = UP     | 1 = DOWN  | 2 = LEFT  | 3 = RIGHT
    4 = PICKUP | 5 = TOGGLE
"""

import numpy as np

# ============================================================
# Constantes de celda
# ============================================================
EMPTY       = 0
WALL        = 1
KEY         = 2
BALL        = 3
DOOR_CLOSED = 4
DOOR_OPEN   = 5
GOAL        = 6

# ============================================================
# Acciones
# ============================================================
ACTION_UP     = 0
ACTION_DOWN   = 1
ACTION_LEFT   = 2
ACTION_RIGHT  = 3
ACTION_PICKUP = 4
ACTION_TOGGLE = 5
N_ACTIONS     = 6

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "PICKUP", "TOGGLE"]

MOVES = {
    ACTION_UP:    (0, -1),
    ACTION_DOWN:  (0,  1),
    ACTION_LEFT:  (-1, 0),
    ACTION_RIGHT: (1,  0),
}


class TwoRoomsEnv:
    """
    Ambiente determinista de dos habitaciones.

    Cuadrícula de 9×5 con pared vertical divisoria en x=4.
    La puerta está en (4, 3) — fila inferior de la pared divisoria.

    El agente debe:
      1. Remover la bola en (2,3) → PICKUP sobre esa casilla
      2. Recoger la llave en (3,1) → PICKUP sobre esa casilla
      3. Abrir la puerta en (4,3) → TOGGLE adyacente a ella
      4. Navegar hasta la meta en (6,0)

    El orden de las sub-tareas 1 y 2 depende del aprendizaje del agente.
    """

    WIDTH  = 9
    HEIGHT = 5
    MAX_STEPS = 200

    # Posiciones fijas de los elementos (columna, fila)
    AGENT_START = (0, 2)   # Esquina izquierda, fila central
    KEY_POS     = (3, 1)   # Columna derecha de habitación izq., fila 1
    BALL_POS    = (3, 3)   # Frente a la puerta, fila 3
    DOOR_POS    = (4, 3)   # Pared divisoria, fila 3 (donde hay puerta)
    GOAL_POS    = (6, 0)   # Habitación derecha, fila superior
    WALL_COL    = 4        # Toda la columna 4 es pared excepto la puerta

    def __init__(self):
        self.reset()

    # ================================================================
    # API principal
    # ================================================================

    def reset(self):
        self.agent_pos = list(self.AGENT_START)
        self.has_key        = 0
        self.door_open      = 0
        self.ball_removed   = 0
        self.steps          = 0
        self._key_bonus_given  = False
        self._ball_bonus_given = False
        self._door_bonus_given = False
        return self._get_state()

    def step(self, action):
        """
        Ejecuta una acción y retorna (estado, recompensa, done, info).

        R(s,a,s'):
          +100  → meta alcanzada
          +20   → puerta abierta (1ª vez en el episodio)
          +10   → llave recogida (1ª vez en el episodio)
          +10   → bola removida  (1ª vez en el episodio)
          -1.0  → acción inválida
          -0.1  → costo de vida por paso
          -10   → timeout (>200 pasos)
        """
        self.steps += 1
        reward = -0.1
        done   = False
        info   = {"event": None}

        # ---- Movimiento ----
        if action in MOVES:
            dx, dy = MOVES[action]
            nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy
            if self._is_blocked(nx, ny):
                reward = -1.0
            else:
                self.agent_pos = [nx, ny]

        # ---- PICKUP: el agente debe estar SOBRE la casilla del objeto ----
        elif action == ACTION_PICKUP:
            ax, ay = self.agent_pos
            content = self._cell_content(ax, ay)

            if content == BALL and not self.ball_removed:
                self.ball_removed = 1
                if not self._ball_bonus_given:
                    reward = +10.0
                    self._ball_bonus_given = True
                    info["event"] = "BALL_REMOVED"

            elif content == KEY and not self.has_key:
                self.has_key = 1
                if not self._key_bonus_given:
                    reward = +10.0
                    self._key_bonus_given = True
                    info["event"] = "KEY_PICKED"
            else:
                reward = -1.0

        # ---- TOGGLE: abrir la puerta ----
        elif action == ACTION_TOGGLE:
            if (self._is_adjacent_to_door()
                    and self.has_key
                    and self.ball_removed
                    and not self.door_open):
                self.door_open = 1
                if not self._door_bonus_given:
                    reward = +20.0
                    self._door_bonus_given = True
                    info["event"] = "DOOR_OPENED"
            else:
                reward = -1.0

        # ---- ¿Llegó a la meta? ----
        if tuple(self.agent_pos) == self.GOAL_POS:
            reward += 100.0
            done = True
            info["event"] = "GOAL_REACHED"

        # ---- Timeout ----
        if self.steps >= self.MAX_STEPS and not done:
            reward = -10.0
            done   = True
            info["event"] = "TIMEOUT"

        return self._get_state(), reward, done, info

    # ================================================================
    # Estado
    # ================================================================

    def _get_state(self):
        """
        S = (x, y, tiene_llave, puerta_abierta, bola_removida)
        Tupla hasheable que sirve como clave de la Q-tabla.
        """
        return (
            self.agent_pos[0],
            self.agent_pos[1],
            self.has_key,
            self.door_open,
            self.ball_removed,
        )

    # ================================================================
    # Lógica interna
    # ================================================================

    def _is_wall(self, x, y):
        if x < 0 or x >= self.WIDTH or y < 0 or y >= self.HEIGHT:
            return True
        # Columna 4 es pared divisoria, excepto la celda de la puerta
        if x == self.WALL_COL and (x, y) != self.DOOR_POS:
            return True
        return False

    def _is_blocked(self, x, y):
        if self._is_wall(x, y):
            return True
        # La puerta cerrada bloquea el paso
        if (x, y) == self.DOOR_POS and not self.door_open:
            return True
        return False

    def _cell_content(self, x, y):
        if self._is_wall(x, y):
            return WALL
        if (x, y) == self.KEY_POS and not self.has_key and not self._key_bonus_given:
            return KEY
        if (x, y) == self.BALL_POS and not self.ball_removed:
            return BALL
        if (x, y) == self.DOOR_POS:
            return DOOR_OPEN if self.door_open else DOOR_CLOSED
        if (x, y) == self.GOAL_POS:
            return GOAL
        return EMPTY

    def _is_adjacent_to_door(self):
        ax, ay = self.agent_pos
        dx, dy = self.DOOR_POS
        return abs(ax - dx) + abs(ay - dy) == 1

    # ================================================================
    # Render ASCII
    # ================================================================

    def render(self):
        chars = {
            EMPTY: ".", WALL: "#", KEY: "K", BALL: "B",
            DOOR_CLOSED: "D", DOOR_OPEN: "d", GOAL: "G",
        }
        lines = []
        for y in range(self.HEIGHT):
            row = ""
            for x in range(self.WIDTH):
                if [x, y] == self.agent_pos:
                    row += "Ak" if self.has_key else "A "
                else:
                    row += chars[self._cell_content(x, y)] + " "
            lines.append(row)
        status = (f"  Paso={self.steps} | llave={self.has_key} "
                  f"| puerta={self.door_open} | bola_removida={self.ball_removed}")
        return "\n".join(lines) + "\n" + status


if __name__ == "__main__":
    env = TwoRoomsEnv()
    env.reset()
    print("=== Layout del ambiente ===")
    print(env.render())
    print(f"\nEstado inicial: {env._get_state()}")
    print(f"Acciones disponibles: {N_ACTIONS} ({', '.join(ACTION_NAMES)})")
