from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.spaces import Discrete

from gym_nonogram.env import Nonogram

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError as exc:
    raise SystemExit(
        "Install the SB3 extras first: pip install -e .[sb3,demo]"
    ) from exc


class FlattenActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.size = env.unwrapped.central_grid_size
        self.action_space = Discrete(self.size * self.size * 2)

    def action(self, action):
        value = action % 2
        cell = action // 2
        row = cell // self.size
        col = cell % self.size
        return row, col, value


class RewardHistoryCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True


def make_env():
    return Monitor(FlattenActionWrapper(Nonogram(central_grid_size=5, seed=7)))


def main():
    output_path = Path("docs/assets/sb3_learning_curve.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_env()
    callback = RewardHistoryCallback()
    model = DQN("MlpPolicy", env, verbose=0, learning_starts=50, buffer_size=1_000)
    model.learn(total_timesteps=1_000, callback=callback)

    rewards = callback.episode_rewards
    if rewards:
        x = np.arange(1, len(rewards) + 1)
        plt.figure(figsize=(6, 3))
        plt.plot(x, rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("DQN learning curve")
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Saved {output_path}")
    else:
        print("No completed episodes were recorded.")


if __name__ == "__main__":
    main()
