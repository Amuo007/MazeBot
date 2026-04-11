"""
Evaluation script for the trained Q-learning agent.

This version reports the project metrics more directly: success rate,
average turns, average deaths, average path length, and exploration efficiency.
"""

import os

from learning_agent import QLearningAgent
from environment import MazeEnvironment


def evaluate_qlearning(
    checkpoint_path: str,
    image_path: str = "maze_5_edited.png",
    num_episodes: int = 5,
    max_turns: int = 10000,
    verbose: bool = True,
):
    """Evaluate a trained Q-learning policy in greedy mode."""
    agent = QLearningAgent()
    agent.load_q_table(checkpoint_path)
    agent.epsilon = 0.0

    env = MazeEnvironment(image_path=image_path, maze_size=64)

    successful_turns = []
    successful_path_lengths = []
    episode_deaths = []
    exploration_efficiencies = []
    success_count = 0

    if verbose:
        print(f"Evaluating: {checkpoint_path}")
        print(f"Episodes: {num_episodes}\n")
        print(f"{'Ep':<4} {'Status':<8} {'Turns':<8} {'Deaths':<8} {'Path':<10} {'Explore':<8}")
        print("-" * 60)

    for episode in range(num_episodes):
        env.reset()
        agent.reset_episode(env.start)
        last_result = None

        while env.turns_taken < max_turns:
            actions = agent.plan_turn(last_result)
            result = env.step(actions)
            last_result = result

            if result.is_goal_reached:
                success_count += 1
                break

        stats = env.get_episode_stats()
        path_length = len(env.cells_visited)
        exploration_efficiency = len(env.unique_cells) / path_length if path_length else 0.0

        if stats["goal_reached"]:
            successful_turns.append(stats["turns_taken"])
            successful_path_lengths.append(path_length)

        episode_deaths.append(stats["deaths"])
        exploration_efficiencies.append(exploration_efficiency)

        if verbose:
            status = "SUCCESS" if stats["goal_reached"] else "TIMEOUT"
            print(
                f"{episode + 1:<4} {status:<8} {stats['turns_taken']:<8} {stats['deaths']:<8} "
                f"{path_length:<10} {exploration_efficiency:<8.3f}"
            )

    success_rate = success_count / num_episodes if num_episodes else 0.0
    avg_turns = sum(successful_turns) / len(successful_turns) if successful_turns else 0.0
    avg_deaths = sum(episode_deaths) / len(episode_deaths) if episode_deaths else 0.0
    avg_path_length = (
        sum(successful_path_lengths) / len(successful_path_lengths)
        if successful_path_lengths
        else 0.0
    )
    avg_exploration_efficiency = (
        sum(exploration_efficiencies) / len(exploration_efficiencies)
        if exploration_efficiencies
        else 0.0
    )

    if verbose:
        print("-" * 60)
        print("\nSummary:")
        print(f"  Success Rate: {success_rate * 100:.1f}%")
        print(f"  Average Turns: {avg_turns:.1f}")
        print(f"  Average Deaths: {avg_deaths:.2f}")
        print(f"  Average Path Length: {avg_path_length:.1f}")
        print(f"  Exploration Efficiency: {avg_exploration_efficiency:.3f}")
        if not successful_turns:
            print("  Note: no successful episodes were completed, so turn and path averages are zero.")

    return {
        "success_rate": success_rate,
        "avg_turns": avg_turns,
        "avg_deaths": avg_deaths,
        "avg_path_length": avg_path_length,
        "exploration_efficiency": avg_exploration_efficiency,
    }


if __name__ == "__main__":
    if os.path.exists("q_agent_final.pkl"):
        print("=" * 60)
        print("EVALUATING FINAL MODEL")
        print("=" * 60)
        evaluate_qlearning("q_agent_final.pkl", num_episodes=5, verbose=True)
    else:
        print("No trained model found at q_agent_final.pkl")
        print("Please run train_qlearning.py first.")
