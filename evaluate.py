"""Evaluación del agente entrenado en modo greedy."""

import argparse
import numpy as np
from environment import TwoRoomsEnv, ACTION_NAMES
from agent import QLearningAgent


def evaluate(qtable_path="qtable.pkl", n_episodes=10, render=False):
    env = TwoRoomsEnv()
    agent = QLearningAgent()
    agent.load_qtable(qtable_path)
    agent.epsilon = 0.0

    all_r, all_s, all_ok = [], [], []

    for i in range(n_episodes):
        state = env.reset()
        total_r, done, success = 0.0, False, 0
        if render and i == 0:
            print("\n=== ESTADO INICIAL ===")
            print(env.render())

        while not done:
            a = agent.select_action(state, greedy=True)
            state, r, done, info = env.step(a)
            total_r += r
            event = info.get("event")
            if event == "GOAL_REACHED":
                success = 1
            if render and i == 0:
                ev_str = f"  *** {event} ***" if event else ""
                print(f"\nPaso {env.steps:2d} | {ACTION_NAMES[a]:6s} | "
                      f"R={r:+7.2f}{ev_str}")
                print(env.render())

        all_r.append(total_r)
        all_s.append(env.steps)
        all_ok.append(success)

    print("\n" + "=" * 50)
    print("RESUMEN DE EVALUACIÓN (modo greedy, ε=0)")
    print("=" * 50)
    print(f"  Episodios:           {n_episodes}")
    print(f"  Tasa de éxito:       {np.mean(all_ok)*100:.1f}%")
    print(f"  Recompensa prom:     {np.mean(all_r):.2f} ± {np.std(all_r):.2f}")
    print(f"  Pasos promedio:      {np.mean(all_s):.1f}")
    print(f"  Estados en Q-tabla:  {len(agent.Q)}")
    print("=" * 50)
    return all_r, all_s, all_ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--qtable", type=str, default="qtable.pkl")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--render", action="store_true")
    args = p.parse_args()
    evaluate(args.qtable, args.episodes, args.render)
