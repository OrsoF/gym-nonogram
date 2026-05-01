from gym_nonogram.env import Nonogram


def main():
    env = Nonogram(central_grid_size=5, seed=7, max_step=50)
    observation, info = env.reset()
    terminated = False
    truncated = False

    print("Enter moves as: row col value")
    print("Rows and columns are zero-based. Value is 0 or 1.")

    while not terminated and not truncated:
        env.render()
        raw_action = input("move> ").strip()
        if raw_action.lower() in {"q", "quit", "exit"}:
            break

        try:
            row, col, value = [int(part) for part in raw_action.split()]
        except ValueError:
            print("Invalid move. Example: 2 3 1")
            continue

        observation, reward, terminated, truncated, info = env.step((row, col, value))
        print(f"reward={reward}")

    env.render()
    print(f"terminated={terminated}, truncated={truncated}")


if __name__ == "__main__":
    main()
