import numpy as np
from board import Board
from stratego_agent import Stratego
from src.agents import SearchAgent

def test_stratego_player_constructor():
    assert Stratego(2,4).IlllIIllIII==1
    assert Stratego(1,4).IlllIIllIII==2

def test_horizontal_capture_no_vertical_capture():
    b=Board(6,7)
    b.add_tile(0,1);b.add_tile(1,2);b.add_tile(2,1)
    assert list(b._grid[-1,:3])==[1,1,1]
    b2=Board(6,7)
    b2.add_tile(0,1);b2.add_tile(0,2);b2.add_tile(0,1)
    assert list(b2._grid[-3:,0])==[1,2,1]

def test_capture_second_call_idempotent_simple_case():
    b=Board(6,7);b.add_tile(0,1);b.add_tile(1,2);b.add_tile(2,1)
    old=b._grid.copy();winner=b.winner;count=b._empty_spaces
    b._add_connectedd_tiles((5,2))
    assert np.array_equal(old,b._grid) and b.winner==winner and b._empty_spaces==count

def test_minimax_pruned_matches_unpruned_values():
    b=Board(6,7)
    for p,col in [(1,3),(2,2),(1,3),(2,2),(1,4),(2,1)]:
        b.add_tile(col,p)
        if b.is_final():break
    assert not b.is_final()
    a=SearchAgent(1,3,pruning=False,heuristic='windows',random_ties=False)
    c=SearchAgent(1,3,pruning=True,heuristic='windows',random_ties=False)
    a.next_action(b);c.next_action(b)
    assert a.last_value==c.last_value
    assert c.nodes<=a.nodes

def test_capturing_winning_move():
    b=Board(6,7)
    for col,p in [(0,1),(1,2),(2,1)]:b.add_tile(col,p)
    assert b._grid[5,1]==1
    assert b.winner==0
