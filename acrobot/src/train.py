import argparse
import csv
import pickle
import time
from pathlib import Path
import gymnasium as gym
import numpy as np
from .discretizer import Discretizer
from .rl import TabularAgent

ROOT = Path(__file__).resolve().parents[1]

def train(algorithm="q", mode="angles", bins=8, episodes=300, seed=42,
          alpha=.1, gamma=.99, eps_min=.05, eps_decay=.999, planning_steps=10,
          name="experiment"):
    if algorithm not in ("q", "dyna"):
        raise ValueError("algorithm debe ser q o dyna")
    env = gym.make("Acrobot-v1", render_mode=None, max_episode_steps=700)
    discretizer = Discretizer(mode, bins)
    agent = TabularAgent(n_actions=env.action_space.n, alpha=alpha, gamma=gamma,
                         planning_steps=planning_steps if algorithm == "dyna" else 0,
                         seed=seed)
    rows = []
    start = time.perf_counter()
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            state = discretizer.transform(obs)
            epsilon = max(eps_min, eps_decay ** episode)
            total_reward = 0.0
            for step in range(1, 701):
                action = agent.act(state, epsilon)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = discretizer.transform(next_obs)
                agent.learn(state, action, reward, next_state, terminated, truncated)
                total_reward += float(reward)
                state = next_state
                if terminated or truncated:
                    break
            rows.append({"episode": episode+1, "return": total_reward,
                         "steps": step, "success": int(terminated),
                         "epsilon": epsilon, "states_visited": len(agent.q),
                         "elapsed_seconds": time.perf_counter()-start})
    finally:
        env.close()
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"models").mkdir(exist_ok=True)
    with (ROOT/"results"/f"{name}.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    artifact = {"algorithm": algorithm, "discretizer": discretizer,
                "q": agent.export_q(), "config": {"mode": mode, "bins": bins,
                "episodes": episodes, "seed": seed, "alpha": alpha,
                "gamma": gamma, "eps_min": eps_min, "eps_decay": eps_decay,
                "planning_steps": agent.planning_steps}}
    with (ROOT/"models"/f"{name}.pkl").open("wb") as file:
        pickle.dump(artifact, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{name}: reward últimos 50={np.mean([r['return'] for r in rows[-50:]]):.1f}, "
          f"éxito últimos 50={np.mean([r['success'] for r in rows[-50:]]):.1%}, "
          f"estados={len(agent.q)}, tiempo={time.perf_counter()-start:.1f}s")
    return artifact, rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", choices=["q", "dyna"], default="q")
    p.add_argument("--mode", choices=["six", "angles", "nonuniform"], default="angles")
    p.add_argument("--bins", type=int, default=8)
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=.1)
    p.add_argument("--gamma", type=float, default=.99)
    p.add_argument("--eps-min", type=float, default=.05)
    p.add_argument("--eps-decay", type=float, default=.999)
    p.add_argument("--planning-steps", type=int, default=10)
    p.add_argument("--name", default="experiment")
    args = vars(p.parse_args())
    if args["episodes"] < 1: p.error("--episodes debe ser >= 1")
    train(**args)

if __name__ == "__main__": main()
