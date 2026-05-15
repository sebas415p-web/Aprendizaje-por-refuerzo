"""Entrenamiento del agente Q-learning."""

import argparse
import json
import numpy as np
from environment import TwoRoomsEnv
from agent import QLearningAgent


def train(episodes=10000, save_path="qtable.pkl", log_path="training_log.json"):
    env   = TwoRoomsEnv()
    agent = QLearningAgent(alpha=0.1, gamma=0.99, epsilon=1.0,
                           epsilon_min=0.05, epsilon_decay=0.9995)

    rewards_h, steps_h, success_h = [], [], []

    print("=" * 65)
    print("ENTRENAMIENTO - Q-learning en Ambiente de Dos Habitaciones")
    print("=" * 65)
    print(f"Episodios={episodes} | α={agent.alpha} | γ={agent.gamma} | "
          f"ε: {agent.epsilon}→{agent.epsilon_min} | decay={agent.epsilon_decay}")
    print("-" * 65)

    for ep in range(episodes):
        state = env.reset()
        ep_r  = 0.0
        done  = False
        success = 0

        while not done:
            a = agent.select_action(state)
            ns, r, done, info = env.step(a)
            agent.update(state, a, r, ns, done)
            state = ns
            ep_r += r
            if info.get("event") == "GOAL_REACHED":
                success = 1

        agent.decay_epsilon()
        rewards_h.append(ep_r)
        steps_h.append(env.steps)
        success_h.append(success)

        if (ep + 1) % 500 == 0:
            w = 500
            print(f"Ep {ep+1:5d} | R̄={np.mean(rewards_h[-w:]):7.2f} | "
                  f"pasos={np.mean(steps_h[-w:]):5.1f} | "
                  f"éxito={np.mean(success_h[-w:])*100:5.1f}% | "
                  f"ε={agent.epsilon:.4f} | |Q|={len(agent.Q)}")

    print("-" * 65)
    agent.save_qtable(save_path)
    agent.export_qtable_readable("qtable_readable.json")

    with open(log_path, "w") as f:
        json.dump({"episodes": episodes,
                   "rewards": rewards_h,
                   "steps": steps_h,
                   "success": success_h}, f)
    print(f"[SAVE] Log guardado en '{log_path}'")

    print(f"\nÚltimos 1000 ep → R̄={np.mean(rewards_h[-1000:]):.2f} | "
          f"pasos={np.mean(steps_h[-1000:]):.1f} | "
          f"éxito={np.mean(success_h[-1000:])*100:.1f}%")
    return agent


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10000)
    p.add_argument("--save", type=str, default="qtable.pkl")
    args = p.parse_args()
    train(args.episodes, args.save)
