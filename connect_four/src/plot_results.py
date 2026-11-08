import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('csv'); args=p.parse_args()
    df=pd.read_csv(args.csv);dest=Path(args.csv).with_suffix('.png')
    if {'pruning','ordering','nodes'}.issubset(df.columns):
        df['variant']=df.apply(lambda r: 'Sin poda' if not r.pruning else ('Poda + orden' if r.ordering else 'Poda'),axis=1)
        avg=df.groupby('variant',sort=False)['nodes'].mean()
        ax=avg.plot.bar(figsize=(8,5));ax.set_ylabel('Nodos medios expandidos');ax.set_title('Impacto de poda alfa-beta')
    elif {'outcome','a_player'}.issubset(df.columns):
        tab=pd.crosstab(df['a_player'],df['outcome']);ax=tab.plot.bar(figsize=(8,5))
        ax.set_ylabel('Partidas');ax.set_title('Resultados por jugador inicial')
    else:raise ValueError('CSV incompatible')
    plt.tight_layout();plt.savefig(dest,dpi=200);plt.close();print('Guardado:',dest)
if __name__=='__main__':main()
