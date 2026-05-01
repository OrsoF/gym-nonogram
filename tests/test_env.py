import numpy as np

from gym_nonogram.env import (
    Nonogram,
    generate_central_grid,
    generate_count,
    generate_left_top_grids,
)


def test_generate_count():
    assert generate_count([0, 1, 1, 0, 1, 0, 1, 1, 1]) == [2, 1, 3]


def test_generate_central_grid_has_one_per_row():
    grid = generate_central_grid(5, 0, seed=1)

    assert grid.shape == (5, 5)
    assert np.all(grid.sum(axis=1) >= 1)


def test_generate_left_top_grids_shape():
    central_grid = np.array(
        [
            [1, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
        ]
    )
    left_grid, top_grid = generate_left_top_grids(central_grid)

    assert left_grid.shape == (3, 2)
    assert top_grid.shape == (2, 3)


def test_env_reset_and_step():
    env = Nonogram(central_grid_size=3, seed=1, max_step=10)
    observation, info = env.reset()

    assert observation.shape == (5, 5)
    assert env.observation_space.contains(observation)
    assert info == {}

    action = (0, 0, int(env.solution_grid[env.small_grid_size, env.small_grid_size]))
    observation, reward, terminated, truncated, info = env.step(action)

    assert reward == 100
    assert not terminated
    assert not truncated
    assert info == {}
    assert observation[env.small_grid_size, env.small_grid_size] > 0


def test_env_truncates_at_max_step():
    env = Nonogram(central_grid_size=3, seed=1, max_step=1)
    env.reset()

    observation, reward, terminated, truncated, info = env.step((0, 0, 1))

    assert env.observation_space.contains(observation)
    assert not terminated
    assert truncated
