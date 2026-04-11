"""
Playback helper for a trained Q-learning agent.
Loads a saved Q-table and reuses the existing visualizer.
"""

import os

from environment import MazeEnvironment
from learning_agent import QLearningAgent
from visualizer import animate_episode

IMAGE_PATH = "maze_5_edited.png"
ANIMATION_FRAME_MS = 100
CHECKPOINT_PATH = "q_agent_final.pkl"


def play_trained_agent(
    checkpoint_path: str = CHECKPOINT_PATH,
    image_path: str = IMAGE_PATH,
    frame_ms: int = ANIMATION_FRAME_MS,
):
    env = MazeEnvironment(image_path=image_path, maze_size=64)
    agent = QLearningAgent(start=env.start)
    agent.load_q_table(checkpoint_path)
    agent.reset_episode(env.start)

    animate_episode(env, agent, max_turns=10000, frame_ms=frame_ms)


if __name__ == "__main__":
    if os.path.exists(CHECKPOINT_PATH):
        play_trained_agent(CHECKPOINT_PATH)
    else:
        print(f"No trained model found at {CHECKPOINT_PATH}")
        print("Please run train_qlearning.py first.")
