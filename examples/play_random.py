import gymnasium as gym
import gym_nonogram


env = gym.make("Nonogram-v0")
observation, info = env.reset()

terminated = False
truncated = False

while not terminated and not truncated:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(action, reward)
