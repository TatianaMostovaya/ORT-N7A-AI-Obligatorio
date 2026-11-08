"""Evaluation functions for the provided capture-enabled Connect Four Board."""
from dataclasses import dataclass
import numpy as np

DIRECTIONS = ((0,1),(1,0),(1,1),(-1,1))
CAPTURE_DIRECTIONS = ((0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1))

@dataclass(frozen=True)
class Weights:
    windows2: float = 2.0
    windows3: float = 14.0
    center: float = 3.0
    material: float = 2.0
    capture: float = 8.0
    immediate: float = 120.0

PRESETS = {
    'windows': Weights(2,14,0,0,0,0),
    'center': Weights(0,0,3,0,0,0),
    'material': Weights(0,0,0,2,0,0),
    'capture': Weights(0,0,0,0,8,0),
    'immediate': Weights(0,0,0,0,0,120),
    'balanced': Weights(),
    'tactical': Weights(2,18,3,3,12,200),
}

def _windows(board):
    h,w=board.heigth,board.length
    g=board._grid
    for r in range(h):
        for c in range(w):
            for dr,dc in DIRECTIONS:
                er,ec=r+3*dr,c+3*dc
                if 0<=er<h and 0<=ec<w:
                    yield [int(g[r+k*dr,c+k*dc]) for k in range(4)]

def _landings(board):
    """Map column -> next playable row according to provided add_tile gravity."""
    out={}
    for c in board.get_posible_actions():
        empty=np.flatnonzero(board._grid[:,c]==0)
        if len(empty):out[c]=int(empty[-1])
    return out

def _capture_potential(board,player,landings):
    g=board._grid; h,w=g.shape; enemy=3-player; score=0
    for c,r in landings.items():
        for dr,dc in CAPTURE_DIRECTIONS:
            ar,ac=r+dr,c+dc
            br,bc=r+2*dr,c+2*dc
            if 0<=br<h and 0<=bc<w and 0<=ar<h and 0<=ac<w:
                if g[ar,ac]==enemy and g[br,bc]==player:
                    score+=1
    return score

def _immediate_wins(board,player):
    """Count playable columns yielding a winner, including captures. Explicitly expensive."""
    result=0
    for c in board.get_posible_actions():
        child=board.clone(); child.add_tile(c,player)
        if child.winner==player:result+=1
    return result

def evaluate(board,player,weights='balanced'):
    if isinstance(weights,str):weights=PRESETS[weights]
    if board.winner:
        return 1_000_000 if board.winner==player else -1_000_000
    if board.is_full():return 0.0
    enemy=3-player; g=board._grid
    score=0.0
    if weights.windows2 or weights.windows3:
        for cells in _windows(board):
            a=cells.count(player); b=cells.count(enemy)
            if a and b:continue
            if a==2:score+=weights.windows2
            elif a==3:score+=weights.windows3
            if b==2:score-=weights.windows2
            elif b==3:score-=weights.windows3
    if weights.center:
        mid=(board.length-1)/2
        for c in range(board.length):
            factor=board.length/2-abs(c-mid)
            score+=weights.center*factor*(np.count_nonzero(g[:,c]==player)-np.count_nonzero(g[:,c]==enemy))
    if weights.material:
        score+=weights.material*(np.count_nonzero(g==player)-np.count_nonzero(g==enemy))
    if weights.capture:
        landings=_landings(board)
        score+=weights.capture*(_capture_potential(board,player,landings)-_capture_potential(board,enemy,landings))
    if weights.immediate:
        score+=weights.immediate*(_immediate_wins(board,player)-_immediate_wins(board,enemy))
    return float(score)
