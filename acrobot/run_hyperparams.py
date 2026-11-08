
import subprocess
import sys
from pathlib import Path

SEEDS = [42, 123, 2026]

BASE = {
    "alpha": 0.1,
    "gamma": 0.99,
    "eps_decay": 0.999,
    "eps_min": 0.05,
    "episodes": 5000,
}

EXPERIMENTS = {
    "alpha_005": {"alpha": 0.05},
    "alpha_02": {"alpha": 0.2},
    "alpha_05": {"alpha": 0.5},
    "gamma_095": {"gamma": 0.95},
    "gamma_0999": {"gamma": 0.999},
    "decay_0995": {"eps_decay": 0.995},
    "decay_09995": {"eps_decay": 0.9995},
    "epsmin_001": {"eps_min": 0.01},
    "epsmin_01": {"eps_min": 0.1},
    "episodes_2500": {"episodes": 2500},
    "episodes_10000": {"episodes": 10000},
}

Path("results/hyperparams").mkdir(
    parents=True, exist_ok=True
)

for experiment_name, overrides in EXPERIMENTS.items():
    config = {**BASE, **overrides}

    for seed in SEEDS:
        name = f"hp_{experiment_name}_seed{seed}"

        print(f"\n=== {name} ===", flush=True)

        train_cmd = [
            sys.executable, "-m", "src.train",
            "--algorithm", "q",
            "--mode", "angles",
            "--bins", "8",
            "--episodes", str(config["episodes"]),
            "--seed", str(seed),
            "--alpha", str(config["alpha"]),
            "--gamma", str(config["gamma"]),
            "--eps-decay", str(config["eps_decay"]),
            "--eps-min", str(config["eps_min"]),
            "--name", name,
        ]

        model = Path("models") / f"{name}.pkl"
        csv = Path("results") / f"{name}.csv"

        if not model.exists() or not csv.exists():
            subprocess.run(train_cmd, check=True)

        json_path = (
            Path("results/hyperparams")
            / f"eval_{name}.json"
        )

        if not json_path.exists():
            eval_cmd = [
                sys.executable, "-m", "src.evaluate",
                str(model),
                "--episodes", "100",
                "--seed", "10000",
            ]

            with json_path.open("w") as output:
                subprocess.run(
                    eval_cmd,
                    stdout=output,
                    check=True,
                )

print("\nExperimentos completados.")
