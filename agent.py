"""Agente Q-learning tabular con guardado/carga de la Q-tabla."""

import pickle
import random
import json
from collections import defaultdict
import numpy as np
from environment import N_ACTIONS, ACTION_NAMES


class QLearningAgent:
    def __init__(self, n_actions=N_ACTIONS, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995):
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = defaultdict(lambda: np.zeros(n_actions))

    def select_action(self, state, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        q = self.Q[state]
        max_q = np.max(q)
        best = [a for a in range(self.n_actions) if q[a] == max_q]
        return random.choice(best)

    def update(self, s, a, r, s_next, done):
        target = r if done else r + self.gamma * np.max(self.Q[s_next])
        self.Q[s][a] += self.alpha * (target - self.Q[s][a])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ---- Guardar Q-tabla ----
    def save_qtable(self, path="qtable.pkl"):
        data = {
            "Q": dict(self.Q),
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "n_actions": self.n_actions,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[SAVE] Q-tabla guardada en '{path}' ({len(self.Q)} estados)")

    # ---- Cargar Q-tabla ----
    def load_qtable(self, path="qtable.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))
        for k, v in data["Q"].items():
            self.Q[k] = np.array(v)
        self.alpha         = data.get("alpha", self.alpha)
        self.gamma         = data.get("gamma", self.gamma)
        self.epsilon       = data.get("epsilon", self.epsilon)
        self.epsilon_min   = data.get("epsilon_min", self.epsilon_min)
        self.epsilon_decay = data.get("epsilon_decay", self.epsilon_decay)
        print(f"[LOAD] Q-tabla cargada desde '{path}' ({len(self.Q)} estados)")

    # ---- Exportar legible ----
    def export_qtable_readable(self, path="qtable_readable.json"):
        readable = {}
        for state, values in sorted(self.Q.items()):
            k = f"x={state[0]},y={state[1]},llave={state[2]},puerta={state[3]},bola={state[4]}"
            readable[k] = {ACTION_NAMES[a]: round(float(values[a]), 3)
                           for a in range(self.n_actions)}
        with open(path, "w") as f:
            json.dump(readable, f, indent=2)
        print(f"[EXPORT] Q-tabla legible guardada en '{path}'")
