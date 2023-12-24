from gym import Env
from gym.spaces import Discrete, Box, Tuple
import numpy as np
from scipy.stats import bernoulli

import numpy as np

def remove_zeros(line : list) -> list:
    '''
    Remove all the zeros at the beginning of a list.
    '''
    if not len(line): # If line == []
        return line
    elif line[0]: # If line[0] == 1
        return line
    else:
        return remove_zeros(line[1:])

def count_ones_and_remove(line : list, count=0):
    """
    Count the number of 1 at the beginning of the list.
    """
    if not len(line): # If line == []
        return line, count
    elif not line[0]: # If line[0] == 0
        return line, count
    else: # line[0] is equal to 1
        return count_ones_and_remove(line[1:], count+1)

def generate_count(line : list) -> list:
    '''
    Count the number of ones in a row for each series of 1.
    '''
    result = []
    while len(line):
        line = remove_zeros(line)
        line, count = count_ones_and_remove(line)
        if count:
            result.append(count)
    return result

def generate_left_top_grids(central_grid : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    For a given central grid of zeros and ones, generate the left 
    and upper tables with inputs of the nonogram.
    """
    grid_size = central_grid.shape[0]

    left_grid = np.zeros((grid_size, grid_size//2+1))
    top_grid = np.zeros((grid_size//2+1, grid_size))

    for i in range(grid_size):
        res_horizontal = generate_count(central_grid[i])
        left_grid[i][:len(res_horizontal)] = res_horizontal
        res_vertical = generate_count(central_grid[:, i])
        top_grid[:, i][:len(res_vertical)] = res_vertical

    return left_grid, top_grid

def left_top_central_grid_to_extended_grid(left_grid : np.ndarray, 
                                           top_grid : np.ndarray, 
                                           central_grid : np.ndarray) -> np.ndarray:
    central_grid_size = central_grid.shape[0]
    small_grid_size = central_grid_size//2+1
    extended_grid_size = central_grid_size + small_grid_size
    assert left_grid.shape == (central_grid_size, small_grid_size) 
    assert top_grid.shape == (small_grid_size, central_grid_size) 
    extended_grid = np.zeros((extended_grid_size, extended_grid_size))
    extended_grid[small_grid_size:, small_grid_size:] = central_grid
    extended_grid[small_grid_size:, :small_grid_size] = left_grid
    extended_grid[:small_grid_size, small_grid_size:] = top_grid
    return extended_grid

def state_to_tuple(mat, n):

    first_column = mat[:n*(n//2+1)].reshape((n, (n//2+1)))
    second_column = mat[n*(n//2+1):2*n*(n//2+1)].reshape((n, (n//2+1)))
    grid = mat[2*n*(n//2+1):].reshape((n, n))

    return first_column, second_column, grid

def generate_central_grid(size : int, proportion : float, seed : float) -> np.ndarray:
    """
    This function generate a random grid that determines a drawing in nonogram.
    """
    assert size > 0, "Size should be positive."
    assert 0 <= proportion <= 1, "Proposition of non zero entries should be between 0 and 1."
    np.random.seed(seed)
    central_grid = bernoulli.rvs(proportion, size=(size, size))
    for row in central_grid:
        if np.max(row) == 0:
            row[np.random.randint(0, size)] = 1
    return central_grid

def generate_problem(size : int, proportion : float, seed : float = 0) -> np.ndarray:
    '''
    Generate a full grid with left, top and drawing for nonogram.
    ''' 
    central_grid = generate_central_grid(size, proportion, seed)
    left_grid, top_grid = generate_left_top_grids(central_grid)
    extended_grid = left_top_central_grid_to_extended_grid(left_grid, top_grid, central_grid)
    return extended_grid


def make_full_row(row, n):
    full_row = np.zeros((n))
    i = 0
    for elem in row:
        full_row[i:i+elem] = 1
        i += 1+elem
    return full_row

def transform_side_grid(grid, n):
    grid_transformed = np.zeros((n, n))
    grid = grid.astype(int)
    for i, row in enumerate(grid):
        grid_transformed[i] = make_full_row(row, n)
    return grid_transformed

class Nonogram(Env):
    def __init__(self, central_grid_size=5, seed = 0, max_step=100):
        self.seed = seed
        self.max_step = max_step

        self.central_grid_size = central_grid_size
        self.small_grid_size = self.central_grid_size//2 +1
        self.extended_grid_size = self.central_grid_size + self.small_grid_size

        self.action_space = Tuple((Discrete(central_grid_size), Discrete(central_grid_size), Discrete(2))) # First coord, second coord, action
        self.observation_space = Box(low=np.zeros((self.extended_grid_size, self.extended_grid_size)), 
                                     high=2*np.ones((self.extended_grid_size, self.extended_grid_size)))


    def render(self):
        print(self.current_grid)


    def reset(self, central_grid_density = .5):
        self.solution_grid = generate_problem(self.central_grid_size, central_grid_density, self.seed)

        self.state = self.solution_grid.copy()
        self.state[self.small_grid_size:, self.small_grid_size:] = 0

        self.number_of_loss = 0
        self.steps_played = 0

        return self.state, {}
    
    def step(self, action):
        row_coord, col_coord, action_choosed = action

        if self.state[self.small_grid_size + row_coord, self.small_grid_size + col_coord] > 0:
            # Already played this square
            obs = self.state
            reward = 0
            terminated = False
            truncated = False
            info = {}

        elif self.solution_grid[self.small_grid_size + row_coord, self.small_grid_size + col_coord] == action_choosed:
            # Action is ok
            obs = self.state
            reward = 100
            terminated = False
            truncated = False
            info = {}

        else:
            # Action is wrong
            obs = self.state
            reward = -20
            terminated = False
            truncated = False
            info = {}
            self.number_of_loss += 1
        
        # Update grid
        self.state[self.small_grid_size + row_coord, self.small_grid_size + col_coord] = self.solution_grid[self.small_grid_size + row_coord, self.small_grid_size + col_coord] + 1

        if self.number_of_loss == 3:
            # Maxmimum number of loss
            obs = self.state
            reward = -100
            terminated = True
            truncated = False
            info = {}

        if self.steps_played == self.max_step:
            # Max step reached
            obs = self.state
            reward = 0
            terminated = False
            truncated = True
            info = {}

        if np.min(self.state[:self.central_grid_size]) == 1:
            # End of the grid
            obs = self.state
            reward = 0
            terminated = False
            truncated = True
            info = {}

        return obs, reward, terminated, truncated, info
