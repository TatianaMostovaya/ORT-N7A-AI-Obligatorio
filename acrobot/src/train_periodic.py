import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from .discretizer import Discretizer
from .rl import TabularAgent

ROOT = Path(__file__).resolve().parents[1]
MAX_STEPS = 700


def greedy_evaluation(agent, discretizer, episodes, seed):
    """Evalúa sin exploración, sin llamar a act() ni modificar Q o rng de entrenamiento."""
    rng = np.random.default_rng(seed)
    env = gym.make("Acrobot-v1", render_mode=None, max_episode_steps=MAX_STEPS)
    returns, lengths, successes = [], [], []
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            total = 0.0
            for step in range(1, MAX_STEPS + 1):
                state = discretizer.transform(obs)
                values = agent.q.get(state)  # get() NO crea nuevos estados en defaultdict
                if values is None:
                    action = int(rng.integers(agent.n_actions))
                else:
                    best = np.flatnonzero(values == values.max())
                    action = int(rng.choice(best))
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                if terminated or truncated:
                    break
            returns.append(total)
            lengths.append(step)
            successes.append(int(terminated))
    finally:
        env.close()
    return {
        "eval_mean_return": float(np.mean(returns)),
        "eval_std_return": float(np.std(returns, ddof=1)) if episodes > 1 else 0.0,
        "eval_mean_steps": float(np.mean(lengths)),
        "eval_std_steps": float(np.std(lengths, ddof=1)) if episodes > 1 else 0.0,
        "eval_success_rate": float(np.mean(successes)),
    }


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train(args):
    if args.episodes < 1 or args.eval_every < 1 or args.eval_episodes < 1:
        raise ValueError("episodes, eval-every y eval-episodes deben ser positivos")
    env = gym.make("Acrobot-v1", render_mode=None, max_episode_steps=MAX_STEPS)
    discretizer = Discretizer(args.mode, args.bins)
    agent = TabularAgent(
        n_actions=env.action_space.n,
        alpha=args.alpha,
        gamma=args.gamma,
        planning_steps=args.planning_steps if args.algorithm == "dyna" else 0,
        seed=args.seed,
    )
    training_rows = []
    eval_rows = []
    real_steps = 0
    training_seconds = 0.0
    eval_seconds = 0.0
    try:
        for ep in range(args.episodes):
            started = time.perf_counter()
            obs, _ = env.reset(seed=args.seed + ep)
            state = discretizer.transform(obs)
            epsilon = max(args.eps_min, args.eps_decay ** ep)
            episode_return = 0.0
            for step in range(1, MAX_STEPS + 1):
                action = agent.act(state, epsilon)
                nxt_obs, reward, terminated, truncated, _ = env.step(action)
                nxt_state = discretizer.transform(nxt_obs)
                agent.learn(state, action, reward, nxt_state, terminated, truncated)
                real_steps += 1
                episode_return += float(reward)
                state = nxt_state
                if terminated or truncated:
                    break
            training_seconds += time.perf_counter() - started
            training_rows.append({
                "episode": ep + 1,
                "return": episode_return,
                "steps": step,
                "success": int(terminated),
                "epsilon": epsilon,
                "states_visited": len(agent.q),
                "real_steps": real_steps,
                "training_seconds": training_seconds,
            })
            checkpoint = (ep + 1) % args.eval_every == 0 or ep + 1 == args.episodes
            if checkpoint:
                ev_start = time.perf_counter()
                metrics = greedy_evaluation(agent, discretizer, args.eval_episodes, args.eval_seed)
                eval_seconds += time.perf_counter() - ev_start
                eval_rows.append({
                    "algorithm": args.algorithm,
                    "planning_steps": agent.planning_steps,
                    "seed": args.seed,
                    "episode": ep + 1,
                    "real_steps": real_steps,
                    "training_seconds": training_seconds,
                    "evaluation_seconds_cumulative": eval_seconds,
                    "states_visited": len(agent.q),
                    "eval_episodes": args.eval_episodes,
                    "eval_seed": args.eval_seed,
                    **metrics,
                })
                print(
                    f"{args.name}: ep={ep+1}, real_steps={real_steps}, "
                    f"train_s={training_seconds:.1f}, "
                    f"eval_steps={metrics['eval_mean_steps']:.1f}, "
                    f"success={metrics['eval_success_rate']:.1%}",
                    flush=True,
                )
    finally:
        env.close()

    results = ROOT / "results"
    models = ROOT / "models"
    results.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)
    save_csv(results / f"{args.name}.csv", training_rows)
    save_csv(results / f"{args.name}_greedy.csv", eval_rows)
    artifact = {
        "algorithm": args.algorithm,
        "discretizer": discretizer,
        "q": agent.export_q(),
        "config": {
            "mode": args.mode, "bins": args.bins, "episodes": args.episodes,
            "seed": args.seed, "alpha": args.alpha, "gamma": args.gamma,
            "eps_min": args.eps_min, "eps_decay": args.eps_decay,
            "planning_steps": agent.planning_steps,
        },
    }
    with (models / f"{args.name}.pkl").open("wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        "name": args.name,
        "algorithm": args.algorithm,
        "planning_steps": agent.planning_steps,
        "seed": args.seed,
        "episodes": args.episodes,
        "real_steps": real_steps,
        "training_seconds": training_seconds,
        "evaluation_seconds_cumulative": eval_seconds,
        "last_checkpoint": eval_rows[-1],
    }
    (results / f"{args.name}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description="Acrobot Q/Dyna con evaluación greedy periódica")
    p.add_argument("--algorithm", choices=["q", "dyna"], default="q")
    p.add_argument("--mode", choices=["six", "angles", "nonuniform"], default="angles")
    p.add_argument("--bins", type=int, default=8)
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eps-min", type=float, default=0.05)
    p.add_argument("--eps-decay", type=float, default=0.999)
    p.add_argument("--planning-steps", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--eval-seed", type=int, default=20000)
    p.add_argument("--name", required=True)
    train(p.parse_args())


if __name__ == "__main__":
    main()
