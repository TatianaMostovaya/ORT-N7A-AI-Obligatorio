import numpy as np
from src.discretizer import Discretizer
from src.rl import TabularAgent

def test_shape_and_bounds():
    obs = np.array([1., 0., 1., 0., 0., 0.])
    for mode in ["six", "angles", "nonuniform"]:
        disc = Discretizer(mode, bins=8)
        state = disc.transform(obs)
        assert len(state) == (6 if mode == "six" else 4)
        assert all(0 <= v < 8 for v in state)

def test_terminal_no_bootstrap():
    agent = TabularAgent(alpha=1, gamma=.99, seed=42)
    s, nxt = (0,), (1,)
    agent.q[nxt][0] = 100
    agent.learn(s, 1, -1, nxt, terminated=True)
    assert agent.q[s][1] == -1

def test_nonterminal_bootstrap():
    agent = TabularAgent(alpha=1, gamma=.5, seed=42)
    s, nxt = (0,), (1,)
    agent.q[nxt][0] = 10
    agent.learn(s, 1, -1, nxt, terminated=False)
    assert agent.q[s][1] == 4
