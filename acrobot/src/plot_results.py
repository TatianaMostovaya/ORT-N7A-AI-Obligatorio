import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()
    data = pd.read_csv(args.csv)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(data.episode, data["return"], alpha=.2, label="Retorno por episodio")
    ax.plot(data.episode, data["return"].rolling(args.window, min_periods=1).mean(),
            label=f"Promedio móvil ({args.window})")
    ax.set(xlabel="Episodio", ylabel="Retorno acumulado", title="Acrobot: curva de aprendizaje")
    ax.legend()
    fig.tight_layout()
    output = Path(args.csv).with_suffix(".png")
    fig.savefig(output, dpi=160)
    print("Gráfico guardado:", output)
