from agent import MazeAgent
from environment import MazeEnvironment
from visualizer import animate_episode

IMAGE_PATH = "maze_5.png"
FIRE_PHASE_IMAGES = ["maze_5.png", "2.png", "3.png", "4.png"]


def main():
    env = MazeEnvironment(
        image_path=IMAGE_PATH,
        fire_phase_images=FIRE_PHASE_IMAGES,
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

    animate_episode(env, agent, max_turns=10000, frame_ms=90)

    print("\nEpisode stats:")
    print(env.get_episode_stats())


if __name__ == "__main__":
    main()