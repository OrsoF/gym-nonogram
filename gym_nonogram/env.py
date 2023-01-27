from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from scipy.stats import bernoulli

import numpy as np

def remove_zeros(line):
    if not len(line):
        return line
    elif line[0]:
        return line
    else:
        return remove_zeros(line[1:])

def count_ones_and_remove(line, count=0):
    if not(len(line)):
        return line, count
    elif not line[0]:
        return line, count
    else:
        return count_ones_and_remove(line[1:], count+1)

def generate_count(line):
    result = []
    while len(line):
        line = remove_zeros(line)
        line, count = count_ones_and_remove(line)
        if count:
            result.append(count)
    return result

def generate_grids(grid, n):
    left_grid = np.zeros((n, n//2+1))
    top_grid = np.zeros((n//2+1, n))
    for i in range(n):
        res_horizontal = generate_count(grid[i])
        left_grid[i][:len(res_horizontal)] = res_horizontal
        res_vertical = generate_count(grid[:, i])
        top_grid[:, i][:len(res_vertical)] = res_vertical
    return left_grid, top_grid.T

def state_to_tuple(mat, n):
    first_column = mat[:n*(n//2+1)].reshape((n, (n//2+1)))
    second_column = mat[n*(n//2+1):2*n*(n//2+1)].reshape((n, (n//2+1)))
    grid = mat[2*n*(n//2+1):].reshape((n, n))

    return first_column, second_column, grid

def generate_grid(n, proportion):
    np.random.seed(1)
    grid = bernoulli.rvs(proportion, size=(n, n))
    for row in grid:
        if np.max(row) == 0:
            row[np.random.randint(0, n)] = 1
    return grid

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
    def __init__(self, n=5):
        self._n = n
        print(self._n)
        self._action_dim = 2*self._n**2
        self.action_space = Discrete(self._action_dim)
        self.observation_space = Box(low=np.zeros((5*n, n)), high=np.ones((5*n, n)))

          
    def render(self):
        tried_grid = self.state[:self._n]
        empty_cells = self.state[self._n:2*self._n]
        full_cells = self.state[2*self._n:3*self._n]

        print('True grid :\n')
        print(self._true_grid)
        print()

        print('Tries :\n')
        print(tried_grid)
        print()

        print('Empty cells :\n')
        print(empty_cells)
        print()

        print('Full cells :\n')
        print(full_cells)
        print()


    def reset(self):
        np.random.seed(1)
        p = np.sqrt(0.75*np.random.random()+0.25)
        self._true_grid = generate_grid(self._n, p)
        self._left_grid, self._top_grid = generate_grids(self._true_grid, self._n)
        self._tried_boxes = np.zeros((self._n, self._n))
        self._revealed_full = np.zeros((self._n, self._n))
        self._revealed_empty = np.zeros((self._n, self._n))
        self._left_grid = transform_side_grid(self._left_grid, self._n)
        self._top_grid = transform_side_grid(self._top_grid, self._n)
        self._lost_time = 0


        self.state = np.concatenate((self._tried_boxes, self._revealed_empty, self._revealed_full, self._left_grid, self._top_grid))
        return self.state, {}
    
    def step(self, action):
        # Computing the action box and the mode z in (0, 1)
        x, y, z = (action%self._n**2)//self._n, (action%self._n**2)%self._n, action//self._n**2
        info = {'fails' : 0,
                'repeated' : 0,
                'successes' : 0,
                'end_lost' : False,
                'end_success' : False}
        # print(x, y, z)
        if self.state[:self._n][x, y] == 1:
            info['repeated'] += 1
            return self.state, -1, False, info
        else:
            self.state[:self._n][x, y] = 1


        if self._true_grid[x, y] == z:
            info['successes'] += 1
            reward = 100
        else:
            reward = -20
            self._lost_time +=1

        if self._lost_time == 3:
            done = True
            reward = -100
            info['end_lost'] = True
        elif np.min(self.state[:self._n]) == 1:
            done = True
            reward = 200
            info['end_success'] = True
        else:
            done = False

        if self._true_grid[x, y] == 0:
            self.state[self._n:2*self._n][x, y] = 1
        else:
            self.state[2*self._n:3*self._n][x, y] = 1
            
        obs = self.state
        # print('Done')
        # print(reward, done)
        return obs, reward, done, info
