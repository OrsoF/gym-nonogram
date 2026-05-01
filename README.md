# gym-nonogram

This package implements the Nonogram game as a Gymnasium environment.

## Install

```bash
pip install -e .
```

## Usage

```python
import gymnasium as gym
import gym_nonogram

env = gym.make("Nonogram-v0")
observation, info = env.reset()
observation, reward, terminated, truncated, info = env.step((0, 0, 1))
```
