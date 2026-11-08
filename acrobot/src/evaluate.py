import argparse
import json
import pickle
from pathlib import Path
import gymnasium as gym
import numpy as np


def evaluate(model_path, episodes=50, seed=10000):
    with open(model_path, "rb") as file:
        artifact = pickle.load(file)  # Solo cargar archivos .pkl de confianza.
    discretizer, q = artifact["discretizer"], artifact["q"]
    rng = np.random.default_rng(seed)
    env = gym.make("Acrobot-v1", render_mode=None, max_episode_steps=700)
    returns, steps, successes = [], [], []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            score = 0.0
            for t in range(1, 701):
                state = discretizer.transform(obs)
                values = q.get(state)
                if values is None:
                    action = int(rng.integers(env.action_space.n))
                else:
                    best = np.flatnonzero(values == values.max())
                    action = int(rng.choice(best))
                obs, reward, terminated, truncated, _ = env.step(action)
                score += float(reward)
                if terminated or truncated:
                    break
            returns.append(score)
            steps.append(t)
            successes.append(int(terminated))
    finally:
        env.close()
    result = {"model": str(model_path), "episodes": episodes, "seed": seed,
              "mean_return": float(np.mean(returns)),
              "std_return": float(np.std(returns, ddof=1)) if episodes > 1 else 0.,
              "mean_steps": float(np.mean(steps)),
              "std_steps": float(np.std(steps, ddof=1)) if episodes > 1 else 0.,
              "success_rate": float(np.mean(successes))}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10000)
    args = parser.parse_args()
    evaluate(args.model, args.episodes, args.seed)
