import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete, Tuple


def generate_count(line: list) -> list:
    """
    Count the number of ones in a row for each series of 1.
    """
    result = []
    count = 0
    for value in line:
        if value:
            count += 1
        elif count:
            result.append(count)
            count = 0
    if count:
        result.append(count)
    return result


def generate_left_top_grids(central_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    For a given central grid of zeros and ones, generate the left
    and upper tables with inputs of the nonogram.
    """
    grid_size = central_grid.shape[0]

    left_grid = np.zeros((grid_size, grid_size // 2 + 1), dtype=np.int8)
    top_grid = np.zeros((grid_size // 2 + 1, grid_size), dtype=np.int8)

    for i in range(grid_size):
        res_horizontal = generate_count(central_grid[i])
        left_grid[i][: len(res_horizontal)] = res_horizontal
        res_vertical = generate_count(central_grid[:, i])
        top_grid[:, i][: len(res_vertical)] = res_vertical
    return left_grid, top_grid


def left_top_central_grid_to_extended_grid(
    left_grid: np.ndarray, top_grid: np.ndarray, central_grid: np.ndarray
) -> np.ndarray:
    """
    Let us consider a nonogram with a central grid and clue grids.
    We combine it as:
    (left, top, central) -> (0, top \\ left, central)
    """
    central_grid_size = central_grid.shape[0]
    small_grid_size = central_grid_size // 2 + 1
    extended_grid_size = central_grid_size + small_grid_size
    assert left_grid.shape == (central_grid_size, small_grid_size)
    assert top_grid.shape == (small_grid_size, central_grid_size)
    extended_grid = np.zeros((extended_grid_size, extended_grid_size), dtype=np.int8)
    extended_grid[small_grid_size:, small_grid_size:] = central_grid
    extended_grid[small_grid_size:, :small_grid_size] = left_grid
    extended_grid[:small_grid_size, small_grid_size:] = top_grid
    return extended_grid


def generate_central_grid(size: int, proportion: float, seed: int = 0) -> np.ndarray:
    """
    This function generate a random grid that determines a drawing in nonogram.
    """
    assert size > 0, "Size should be positive."
    assert (
        0 <= proportion <= 1
    ), "Proposition of non zero entries should be between 0 and 1."
    rng = np.random.default_rng(seed)
    central_grid = rng.binomial(1, proportion, size=(size, size)).astype(np.int8)
    for row in central_grid:
        if np.max(row) == 0:
            row[rng.integers(0, size)] = 1
    return central_grid


def generate_problem(size: int, proportion: float, seed: int = 0) -> np.ndarray:
    """
    Generate a full grid with left, top and drawing for nonogram.
    """
    central_grid = generate_central_grid(size, proportion, seed)
    left_grid, top_grid = generate_left_top_grids(central_grid)
    extended_grid = left_top_central_grid_to_extended_grid(
        left_grid, top_grid, central_grid
    )
    return extended_grid


class Nonogram(Env):
    def __init__(self, central_grid_size=5, seed=0, max_step=100):
        self.seed = seed
        self.max_step = max_step

        self.central_grid_size = central_grid_size
        self.small_grid_size = self.central_grid_size // 2 + 1
        self.extended_grid_size = self.central_grid_size + self.small_grid_size

        self.action_space = Tuple(
            (Discrete(central_grid_size), Discrete(central_grid_size), Discrete(2))
        )  # First coord, second coord, action
        self.observation_space = Box(
            low=np.zeros(
                (self.extended_grid_size, self.extended_grid_size), dtype=np.int8
            ),
            high=self.central_grid_size
            * np.ones(
                (self.extended_grid_size, self.extended_grid_size), dtype=np.int8
            ),
            dtype=np.int8,
        )

    def render(self):
        print(self.state)

    def reset(self, seed=None, options=None, central_grid_density=0.5):
        super().reset(seed=seed)
        if seed is not None:
            self.seed = seed
        if options and "central_grid_density" in options:
            central_grid_density = options["central_grid_density"]

        self.solution_grid = generate_problem(
            self.central_grid_size, central_grid_density, self.seed
        )

        self.state = self.solution_grid.copy()
        self.state[self.small_grid_size :, self.small_grid_size :] = 0

        self.number_of_loss = 0
        self.steps_played = 0

        return self.state, {}

    def step(self, action):
        row_coord, col_coord, action_choosed = action
        self.steps_played += 1

        if (
            self.state[
                self.small_grid_size + row_coord, self.small_grid_size + col_coord
            ]
            > 0
        ):
            # Already played this square
            obs = self.state
            reward = 0
            terminated = False
            truncated = False
            info = {}

        elif (
            self.solution_grid[
                self.small_grid_size + row_coord, self.small_grid_size + col_coord
            ]
            == action_choosed
        ):
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
        self.state[
            self.small_grid_size + row_coord, self.small_grid_size + col_coord
        ] = (
            self.solution_grid[
                self.small_grid_size + row_coord, self.small_grid_size + col_coord
            ]
            + 1
        )

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

        if np.min(self.state[self.small_grid_size :, self.small_grid_size :]) > 0:
            # End of the grid
            obs = self.state
            reward = 0
            terminated = True
            truncated = False
            info = {}

        return obs, reward, terminated, truncated, info
