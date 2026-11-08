"""Alpha-beta impact measured on identical generated reachable boards."""
import argparse,csv,json,random,time
from pathlib import Path
from board import Board
from .agents import SearchAgent

def generate_positions(n,seed,plies):
    rng=random.Random(seed); positions=[]
    for i in range(n):
        b=Board(6,7)
        for j in range(plies):
            if b.is_final():break
            b.add_tile(rng.choice(b.get_posible_actions()),1+(j%2))
        if not b.is_final():positions.append((i,b,1+(min(plies,42)%2)))
    return positions

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--positions',type=int,default=10)
    p.add_argument('--plies',type=int,default=8)
    p.add_argument('--depth',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--heuristic',default='windows')
    p.add_argument('--name',default='pruning_benchmark')
    args=p.parse_args();rows=[]
    for index,b,player in generate_positions(args.positions,args.seed,args.plies):
        for pruning,ordering in ((False,False),(True,False),(True,True)):
            a=SearchAgent(player,args.depth,algorithm='minimax',pruning=pruning,ordering=ordering,
                          heuristic=args.heuristic,random_ties=False)
            action=a.next_action(b)
            rows.append({'position':index,'pruning':pruning,'ordering':ordering,'player':player,
                         'action':action,'value':a.last_value,'nodes':a.nodes,'leaves':a.leaves,'seconds':a.elapsed})
        # Equal minimax values guarantee pruning did not change solution quality.
        group=rows[-3:]
        if not all(abs(r['value']-group[0]['value'])<1e-6 for r in group):
            raise AssertionError(f'alpha-beta value differs on position {index}')
    Path('results').mkdir(exist_ok=True)
    with open(f'results/{args.name}.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for pruning,ordering in ((False,False),(True,False),(True,True)):
        group=[r for r in rows if (r['pruning'],r['ordering'])==(pruning,ordering)]
        summary.append({'pruning':pruning,'ordering':ordering,'positions':len(group),
                        'mean_nodes':sum(r['nodes'] for r in group)/len(group),
                        'mean_seconds':sum(r['seconds'] for r in group)/len(group)})
    Path(f'results/{args.name}.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
