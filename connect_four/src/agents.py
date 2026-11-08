"""Minimax/alpha-beta and Expectimax compatible with supplied Agent interface."""
import math
import random
import time
from agent import Agent
from .heuristics import evaluate, PRESETS

WIN_SCORE=1_000_000_000

class SearchAgent(Agent):
    def __init__(self,player=1,depth=3,algorithm='minimax',pruning=True,
                 ordering=True,heuristic='balanced',chance='uniform',temperature=50,
                 seed=42,random_ties=True):
        if player not in (1,2):raise ValueError('player must be 1 or 2')
        if algorithm not in ('minimax','expectimax'):raise ValueError('algorithm must be minimax or expectimax')
        if heuristic not in PRESETS:raise ValueError(f'heuristic must be one of {list(PRESETS)}')
        if chance not in ('uniform','softmax'):raise ValueError('chance must be uniform or softmax')
        if depth < 1:raise ValueError('depth must be >=1')
        self.player=player; self.depth=depth; self.algorithm=algorithm
        self.pruning=pruning; self.ordering=ordering; self.heuristic=heuristic
        self.chance=chance; self.temperature=temperature
        self.rng=random.Random(seed);self.random_ties=random_ties
        self.nodes=0;self.leaves=0;self.elapsed=0.0;self.last_value=0.0

    def heuristic_utility(self,board):
        return evaluate(board,self.player,self.heuristic)

    def _score(self,board,depth):
        if board.winner:
            # Larger remaining depth = faster win / slower loss.
            return (WIN_SCORE+depth if board.winner==self.player else -WIN_SCORE-depth)
        if board.is_full():return 0.0
        return self.heuristic_utility(board)

    def _moves(self,board):
        actions=board.get_posible_actions()
        if self.ordering:
            center=(board.length-1)/2
            return sorted(actions,key=lambda c:(abs(c-center),c))
        return actions

    def _chance_probs(self,children):
        if self.chance=='uniform':return [1/len(children)]*len(children)
        if self.temperature<=0:raise ValueError('temperature > 0 required')
        # Rival's preference: maximize their own score = minimize ours.
        vals=[-self._score(child,0)/self.temperature for _,child in children]
        m=max(vals); exps=[math.exp(v-m) for v in vals]; den=sum(exps)
        return [v/den for v in exps]

    def _search(self,board,to_move,depth,alpha,beta):
        self.nodes+=1
        if depth==0 or board.is_final():
            self.leaves+=1
            return self._score(board,depth)
        actions=self._moves(board)
        if not actions:
            self.leaves+=1
            return self._score(board,depth)
        if to_move==self.player:
            value=-math.inf
            for col in actions:
                child=board.clone();child.add_tile(col,to_move)
                value=max(value,self._search(child,3-to_move,depth-1,alpha,beta))
                if self.algorithm=='minimax' and self.pruning:
                    alpha=max(alpha,value)
                    if alpha>=beta:break
            return value
        if self.algorithm=='expectimax':
            children=[]
            for col in actions:
                child=board.clone();child.add_tile(col,to_move)
                children.append((col,child))
            probs=self._chance_probs(children)
            return sum(p*self._search(child,3-to_move,depth-1,alpha,beta)
                       for p,(_,child) in zip(probs,children))
        value=math.inf
        for col in actions:
            child=board.clone();child.add_tile(col,to_move)
            value=min(value,self._search(child,3-to_move,depth-1,alpha,beta))
            if self.pruning:
                beta=min(beta,value)
                if alpha>=beta:break
        return value

    def next_action(self,obs):
        if obs.is_final():raise ValueError('Cannot act on terminal board')
        actions=self._moves(obs)
        if not actions:raise ValueError('No legal moves')
        self.nodes=1;self.leaves=0
        start=time.perf_counter();values=[];alpha=-math.inf;beta=math.inf
        for col in actions:
            child=obs.clone();child.add_tile(col,self.player)
            score=self._search(child,3-self.player,self.depth-1,alpha,beta)
            values.append((col,score))
            # Independent full root window prevents cut-off bounds from being
            # mistaken for exact ties when random tie-breaking is enabled.
        best=max(score for _,score in values)
        ties=[col for col,score in values if abs(score-best)<1e-9]
        choice=self.rng.choice(ties) if self.random_ties else ties[0]
        self.last_value=best;self.elapsed=time.perf_counter()-start
        return choice

class RandomAgent(Agent):
    def __init__(self,player=1,seed=42):
        self.player=player;self.rng=random.Random(seed)
        self.nodes=0;self.elapsed=0.0
    def next_action(self,obs):return self.rng.choice(obs.get_posible_actions())
    def heuristic_utility(self,board):return 0.0
