import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--names", nargs="+", required=True, help="nombres de las ejecuciones")
    p.add_argument("--output-prefix", default="greedy_comparison")
    args = p.parse_args()
    frames = {}
    for name in args.names:
        path = ROOT / "results" / f"{name}_greedy.csv"
        frames[name] = pd.read_csv(path).sort_values("episode")
    axes = [
        ("episode", "episodios", "Episodios entrenados"),
        ("real_steps", "interacciones", "Interacciones reales acumuladas"),
        ("training_seconds", "tiempo", "Tiempo de entrenamiento acumulado (s)"),
    ]
    metrics = [
        ("eval_mean_steps", "pasos", "Pasos medios greedy (menor es mejor)"),
        ("eval_mean_return", "retorno", "Retorno medio greedy (mayor es mejor)"),
        ("eval_success_rate", "exito", "Tasa de éxito greedy (0 a 1)"),
    ]
    for x, suffix, label in axes:
        for metric, metric_suffix, metric_label in metrics:
            fig, ax = plt.subplots(figsize=(11, 6))
            for name, frame in frames.items():
                ax.plot(frame[x], frame[metric], marker="o", markersize=3, label=name)
            ax.set(xlabel=label, ylabel=metric_label,
                   title="Acrobot: evaluación greedy periódica")
            if metric == "eval_success_rate":
                ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            dest = ROOT / "results" / f"{args.output_prefix}_{suffix}_{metric_suffix}.png"
            fig.savefig(dest, dpi=200)
            plt.close(fig)
            print(f"Gráfico guardado: {dest}")

    rows = []
    for name, frame in frames.items():
        final = frame.iloc[-1]
        rows.append({"model": name, "training_seconds": final["training_seconds"],
                     "real_steps": final["real_steps"],
                     "eval_mean_steps": final["eval_mean_steps"],
                     "eval_success_rate": final["eval_success_rate"]})
    summary = pd.DataFrame(rows)
    path = ROOT / "results" / f"{args.output_prefix}_final.csv"
    summary.to_csv(path, index=False)
    print(summary.to_string(index=False))
    print(f"CSV guardado: {path}")


if __name__ == "__main__":
    main()
