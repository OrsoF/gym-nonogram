from gym_nonogram.env import Nonogram


def main():
    env = Nonogram(central_grid_size=5, seed=7, max_step=30)
    observation, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"action={action}, reward={reward}")

    print(f"total_reward={total_reward}")
    print(f"terminated={terminated}, truncated={truncated}")


if __name__ == "__main__":
    main()
