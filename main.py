from agent import MazeAgent
from environment import MazeEnvironment
from visualizer import animate_episode

IMAGE_PATH = "maze_5_edited.png"
ANIMATION_FRAME_MS = 100


def main():
    env = MazeEnvironment(
        image_path=IMAGE_PATH,
        maze_size=64,
    )

    agent = MazeAgent(
        start=env.start,
        goal=env.goal,
        vertical_walls=env.vertical_walls,
        horizontal_walls=env.horizontal_walls,
        obj_matrix=env.obj_matrix,
        teleport_pairs=env.teleport_pairs,
    )

    env.reset()
    agent.reset_episode()

    print(f"Start: {env.start}")
    print(f"Goal : {env.goal}")

    animate_episode(env, agent, max_turns=10000, frame_ms=ANIMATION_FRAME_MS)

    print("\nEpisode stats:")
    print(env.get_episode_stats())


if __name__ == "__main__":
    main()