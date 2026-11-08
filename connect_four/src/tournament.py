"""Noninteractive tournaments: alternating colors and reporting observed winner."""
import argparse,csv,json,random,time
from collections import defaultdict
from pathlib import Path
from board import Board
from stratego_agent import Stratego
from .agents import SearchAgent,RandomAgent

def make_agent(kind,player,depth,seed,heuristic,pruning=True,ordering=True,chance='uniform'):
    if kind=='random':return RandomAgent(player,seed)
    if kind=='stratego':
        # Original Stratego maps constructor 1 -> player 2, constructor 2 -> player 1.
        return Stratego(1 if player==2 else 2,4)
    return SearchAgent(player,depth,kind,pruning,ordering,heuristic,chance,seed=seed)

def play(a_name,b_name,depth,heuristic,seed,swap=False,pruning=True,ordering=True,chance='uniform',opening_random=0):
    board=Board(6,7)
    a_player=2 if swap else 1
    agents={a_player:make_agent(a_name,a_player,depth,seed,heuristic,pruning,ordering,chance),
            3-a_player:make_agent(b_name,3-a_player,depth,seed+10000,heuristic,pruning,ordering,chance)}
    rand=random.Random(seed); moves=0; records=defaultdict(lambda:{'nodes':0,'time':0.0,'moves':0})
    while not board.is_final():
        player=1 if moves%2==0 else 2
        legal=board.get_posible_actions()
        if not legal:break
        agent=agents[player]; t0=time.perf_counter()
        if moves<opening_random:
            action=rand.choice(legal); nodes=0
        else:
            action=agent.next_action(board); nodes=getattr(agent,'nodes',0)
        elapsed=time.perf_counter()-t0
        if action not in legal:raise ValueError(f'{type(agent).__name__} illegal action {action} vs {legal}')
        board.add_tile(action,player)
        row=records[player]; row['moves']+=1;row['time']+=elapsed;row['nodes']+=nodes
        moves+=1
        if moves>42:raise RuntimeError('Board did not finish in 42 moves')
    winner=board.winner
    own=records[a_player]
    return {'seed':seed,'swap':int(swap),'a_player':a_player,'winner':winner,
            'outcome':'win' if winner==a_player else ('draw' if winner==0 else 'loss'),
            'moves':moves,'a_decision_seconds':own['time'],
            'a_mean_decision_seconds':own['time']/max(own['moves'],1),
            'a_expanded_nodes':own['nodes'],
            'a_mean_nodes_per_move':own['nodes']/max(own['moves'],1)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--a',choices=['minimax','expectimax','random','stratego'],default='minimax')
    p.add_argument('--b',choices=['minimax','expectimax','random','stratego'],default='random')
    p.add_argument('--depth',type=int,default=3)
    p.add_argument('--heuristic',default='balanced')
    p.add_argument('--games-per-side',type=int,default=10)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--opening-random',type=int,default=0)
    p.add_argument('--no-pruning',action='store_true')
    p.add_argument('--no-ordering',action='store_true')
    p.add_argument('--chance',choices=['uniform','softmax'],default='uniform')
    p.add_argument('--name',default='tournament')
    args=p.parse_args()
    Path('results').mkdir(exist_ok=True)
    rows=[]
    for swap in (False,True):
        for i in range(args.games_per_side):
            seed=args.seed+i
            row=play(args.a,args.b,args.depth,args.heuristic,seed,swap,
                     not args.no_pruning,not args.no_ordering,args.chance,args.opening_random)
            rows.append(row)
            print(f"{args.a} vs {args.b}: seed={seed} side={row['a_player']} {row['outcome']} moves={row['moves']}",flush=True)
    with open(f'results/{args.name}.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    n=len(rows);summary={'config':vars(args),'games':n,
                         'wins':sum(x['outcome']=='win' for x in rows),
                         'draws':sum(x['outcome']=='draw' for x in rows),
                         'losses':sum(x['outcome']=='loss' for x in rows),
                         'win_rate':sum(x['outcome']=='win' for x in rows)/n,
                         'mean_moves':sum(x['moves'] for x in rows)/n,
                         'mean_decision_seconds':sum(x['a_mean_decision_seconds'] for x in rows)/n,
                         'mean_nodes_per_move':sum(x['a_mean_nodes_per_move'] for x in rows)/n}
    Path(f'results/{args.name}.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
