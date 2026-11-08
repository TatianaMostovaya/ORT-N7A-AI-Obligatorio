import numpy as np
import gymnasium as gym
from gymnasium import spaces
from board import Board

player_one = 1
player_two = 2
x = "x"
y = "y"


class BoardSpace(spaces.Space):
    def __init__(self, heigth, length):
        self.heigth = heigth
        self.length = length
        super().__init__(shape=(heigth, length), dtype=np.int64)

    @property
    def is_np_flattenable(self):
        return False

    def sample(self, mask=None, probability=None):
        board = Board(self.heigth, self.length)
        move_count = self.np_random.integers(self.heigth * self.length + 1)
        current_player = player_one

        for _ in range(move_count):
            if board.is_final():
                break
            actions = board.get_posible_actions()
            action = int(self.np_random.choice(actions))
            board.add_tile(action, current_player)
            current_player = player_two if current_player == player_one else player_one

        return board

    def contains(self, value):
        if not isinstance(value, Board):
            return False

        grid = value._grid
        return (
            value.heigth == self.heigth
            and value.length == self.length
            and isinstance(grid, np.ndarray)
            and grid.shape == self.shape
            and np.issubdtype(grid.dtype, np.integer)
            and np.all((grid >= 0) & (grid <= player_two))
        )


class ConnectFourBaseEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, render_mode="human"):
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.render_mode = render_mode
        self._length = 7
        self._heigth = 6
        self.action_space = spaces.Discrete(self._length)
        self.observation_space = BoardSpace(self._heigth, self._length)
        self._current_player = player_one
        self._reset_grid()
    
    def step(self, action):
        assert (self.action_space.contains(action))
        
        movement_played = self._grid.add_tile(action, self._current_player)
        
        if movement_played:
            self._next_player()
            
        terminated = self._is_done()
        return self._grid, self._get_reward(), terminated, False, {}
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._current_player = player_one
        self._reset_grid()
        return self._grid, {}
    
    def render(self):
        if self.render_mode == "human":
            self._grid.render()
        
    def _get_reward(self):
        return 100 if self._grid.winner != 0 else 0
        
    def _is_done(self):        
        return self._grid.is_final()
        
    def _next_player(self):
        self._current_player = player_one if self._current_player == player_two \
                                    else player_two
        
    def _reset_grid(self):
        self._grid = Board(self._heigth, self._length)