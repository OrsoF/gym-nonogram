from gymnasium.envs.registration import register
from .env import Nonogram

register(
    id='Nonogram-v0',
    entry_point='gym_nonogram.env:Nonogram',
    kwargs={
        'central_grid_size': 5,
    }
)
