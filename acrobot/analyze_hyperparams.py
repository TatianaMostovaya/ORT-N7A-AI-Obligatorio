import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("results")
HP = ROOT / "hyperparams"

EXPERIMENTS = {
    "alpha": ["alpha_005", "alpha_02", "alpha_05"],
    "gamma": ["gamma_095", "gamma_0999"],
    "eps_decay": ["decay_0995", "decay_09995"],
    "eps_min": ["epsmin_001", "epsmin_01"],
    "episodes": ["episodes_2500", "episodes_10000"],
}

BASELINE = {
    42: "eval_q_angles_8_5000.json",
    123: "eval_angles_8_seed123.json",
    2026: "eval_angles_8_seed2026.json",
}

rows = []

# Modelos de referencia
for seed, filename in BASELINE.items():
    path = ROOT / filename
    if not path.exists():
        raise FileNotFoundError(path)

    data = json.loads(path.read_text())

    rows.append({
        "parameter": "baseline",
        "configuration": "base",
        "seed": seed,
        "mean_return": data["mean_return"],
        "mean_steps": data["mean_steps"],
        "std_steps": data["std_steps"],
        "success_rate": data["success_rate"],
    })

# Nuevos experimentos
for parameter, configs in EXPERIMENTS.items():
    for config in configs:
        for seed in [42, 123, 2026]:
            path = HP / f"eval_hp_{config}_seed{seed}.json"

            if not path.exists():
                raise FileNotFoundError(path)

            data = json.loads(path.read_text())

            rows.append({
                "parameter": parameter,
                "configuration": config,
                "seed": seed,
                "mean_return": data["mean_return"],
                "mean_steps": data["mean_steps"],
                "std_steps": data["std_steps"],
                "success_rate": data["success_rate"],
            })

df = pd.DataFrame(rows)

if len(df) != 36:
    raise RuntimeError(f"Se esperaban 36 evaluaciones; hay {len(df)}")

df.to_csv(ROOT / "hyperparams_all.csv", index=False)
df.to_json(
    ROOT / "hyperparams_all.json",
    orient="records",
    indent=2
)

summary = (
    df.groupby(["parameter", "configuration"], as_index=False)
    .agg(
        mean_steps=("mean_steps", "mean"),
        std_between_seeds=("mean_steps", "std"),
        mean_return=("mean_return", "mean"),
        mean_success=("success_rate", "mean"),
    )
)

summary["success_pct"] = summary["mean_success"] * 100

summary = summary.sort_values(
    ["mean_steps", "success_pct"],
    ascending=[True, False]
)

summary.to_csv(ROOT / "hyperparams_summary.csv", index=False)
summary.to_json(
    ROOT / "hyperparams_summary.json",
    orient="records",
    indent=2
)

print("\nRANKING GLOBAL DE CONFIGURACIONES")
print(summary.to_string(index=False))

print("\nMEJOR RESULTADO INDIVIDUAL")
print(
    df.sort_values("mean_steps")
    .head(5)
    .to_string(index=False)
)

# Gráfico global
ordered = summary.sort_values("mean_steps")

fig, ax = plt.subplots(figsize=(12, 7))

ax.barh(
    ordered["configuration"],
    ordered["mean_steps"],
    xerr=ordered["std_between_seeds"],
    capsize=4
)

ax.invert_yaxis()
ax.set_xlabel("Pasos medios de evaluación")
ax.set_title("Q-Learning: sensibilidad a hiperparámetros")
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(
    ROOT / "hyperparams_ranking.png",
    dpi=200
)
plt.close()

# Gráficos individuales por parámetro
for parameter in EXPERIMENTS:
    subset = summary[
        (summary["parameter"] == parameter)
        | (summary["parameter"] == "baseline")
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        subset["configuration"],
        subset["mean_steps"],
        yerr=subset["std_between_seeds"],
        capsize=5
    )

    ax.set_title(f"Q-Learning: impacto de {parameter}")
    ax.set_xlabel("Configuración")
    ax.set_ylabel("Pasos medios de evaluación")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        ROOT / f"hyperparams_{parameter}.png",
        dpi=200
    )
    plt.close()

print("\nAnálisis completado.")
print("CSV, JSON y gráficos PNG guardados en results/")
