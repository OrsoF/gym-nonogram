from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from gym_nonogram.env import Nonogram

ASSET_DIR = Path("docs/assets")


def draw_board(state, small_grid_size, path):
    cell_size = 42
    height, width = state.shape
    image = Image.new("RGB", (width * cell_size, height * cell_size), "white")
    draw = ImageDraw.Draw(image)

    for row in range(height):
        for col in range(width):
            value = int(state[row, col])
            x0 = col * cell_size
            y0 = row * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size

            if row >= small_grid_size and col >= small_grid_size:
                if value == 2:
                    fill = "#222222"
                elif value == 1:
                    fill = "#eeeeee"
                else:
                    fill = "#ffffff"
            else:
                fill = "#f4f0de" if value else "#ffffff"

            draw.rectangle((x0, y0, x1, y1), fill=fill, outline="#999999")
            if value and (row < small_grid_size or col < small_grid_size):
                draw.text((x0 + 16, y0 + 13), str(value), fill="#222222")

    border = small_grid_size * cell_size
    draw.line((0, border, width * cell_size, border), fill="#111111", width=3)
    draw.line((border, 0, border, height * cell_size), fill="#111111", width=3)
    image.save(path)
    return image


def run_rollout(env):
    env.action_space.seed(7)
    state, info = env.reset()
    frames = [state.copy()]
    rewards = []
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        frames.append(state.copy())
        rewards.append(reward)

    return frames, rewards, terminated, truncated


def save_gif(frames, env, path):
    images = [
        draw_board(frame, env.small_grid_size, ASSET_DIR / "_frame.png")
        for frame in frames[:20]
    ]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=350,
        loop=0,
    )
    (ASSET_DIR / "_frame.png").unlink(missing_ok=True)


def save_learning_curve(path):
    rewards = [-220, -160, -80, -60, -20, 40, 80]
    plt.figure(figsize=(6, 3))
    plt.plot(range(1, len(rewards) + 1), rewards, marker="o")
    plt.xlabel("Evaluation")
    plt.ylabel("Reward")
    plt.title("Example learning curve")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def benchmark():
    rows = []
    for size in [3, 5, 7]:
        env = Nonogram(central_grid_size=size, seed=7, max_step=size * size)
        env.action_space.seed(size)
        total_reward = 0
        total_steps = 0
        episodes = 100
        for _ in range(episodes):
            state, info = env.reset()
            terminated = False
            truncated = False
            while not terminated and not truncated:
                state, reward, terminated, truncated, info = env.step(
                    env.action_space.sample()
                )
                total_reward += reward
                total_steps += 1
        rows.append((size, episodes, total_steps / episodes, total_reward / episodes))
    return rows


def save_benchmark(rows, path):
    lines = [
        "# Benchmark",
        "",
        "Random-agent smoke benchmark on generated puzzles.",
        "",
        "| Grid size | Episodes | Mean steps | Mean reward |",
        "| --- | ---: | ---: | ---: |",
    ]
    for size, episodes, mean_steps, mean_reward in rows:
        lines.append(
            f"| {size}x{size} | {episodes} | {mean_steps:.1f} | {mean_reward:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    env = Nonogram(central_grid_size=5, seed=7, max_step=25)
    frames, _rewards, _terminated, _truncated = run_rollout(env)
    solved_state = env.solution_grid.copy()
    solved_state[env.small_grid_size :, env.small_grid_size :] += 1

    draw_board(frames[0], env.small_grid_size, ASSET_DIR / "board_initial.png")
    draw_board(frames[-1], env.small_grid_size, ASSET_DIR / "board_rollout_final.png")
    draw_board(solved_state, env.small_grid_size, ASSET_DIR / "sample_solved.png")
    save_gif(frames, env, ASSET_DIR / "rollout.gif")
    save_learning_curve(ASSET_DIR / "learning_curve_example.png")
    save_benchmark(benchmark(), Path("docs/benchmark.md"))


if __name__ == "__main__":
    main()
