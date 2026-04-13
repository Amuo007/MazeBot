"""
Training script for Q-learning maze agent.

The training loop follows the project spec more closely by using the agent's
plan_turn(last_result) interface, turn-based environment stepping, and rewards
based on observable outcomes and exploration progress rather than goal-distance
shaping.
"""

import os
import time

from learning_agent import QLearningAgent
from environment import MazeEnvironment


def manhattan_distance(a, b) -> int:
    """Compute Manhattan distance between two cells."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def compute_reward(
    result,
    prev_pos,
    goal,
    shaping_weight: float,
) -> float:
    """Compute reward purely based on MDP state transitions."""
    # Standard step penalty to encourage shortest path
    reward = -0.1

    if result.wall_hits > 0:
        reward -= 0.5 * result.wall_hits

    if result.is_dead:
        reward -= 10.0

    if result.is_goal_reached:
        reward += 100.0

    # Potential-style shaping (Markovian)
    prev_dist = manhattan_distance(prev_pos, goal)
    next_dist = manhattan_distance(result.current_position, goal)
    reward += shaping_weight * (prev_dist - next_dist)

    return reward


def train_qlearning(
    image_path: str = "maze_5_edited.png",
    num_episodes: int = 500,
    max_steps: int = 10000,
    log_interval: int = 10,
    checkpoint_interval: int = 50,
    shaping_start: float = 1.0,
    shaping_end: float = 1.0,
):
    """
    Train the Q-Learning agent on the maze.
    
    Args:
        image_path: Path to the maze image
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        log_interval: Print statistics every N episodes
        checkpoint_interval: Save checkpoint every N episodes
    """
    
    os.makedirs("checkpoints", exist_ok=True)

    # Initialize environment and agent
    env = MazeEnvironment(image_path=image_path, maze_size=64)
    agent = QLearningAgent(start=env.start, alpha=0.1, gamma=0.99, epsilon=1.0)
    
    # Tracking metrics
    episode_rewards = []
    episode_steps = []
    episode_successes = []
    episode_death_counts = []
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        # Reset for new episode
        env.reset()
        agent.reset_episode(env.start)
        episode_reward = 0.0
        episode_turns = 0
        deaths_this_episode = 0
        last_result = None

        # Run episode loop using the project turn structure
        while episode_turns < max_steps:
            progress_ratio = episode / max(1, num_episodes - 1)
            shaping_weight = shaping_start + (shaping_end - shaping_start) * progress_ratio

            # Ask agent to plan (this applies last_result and updates its internal state)
            actions = agent.plan_turn(last_result)
            action = actions[0]

            # Capture the state *after* the agent has updated for this turn
            state = agent.state_to_key(agent.current_pos, agent.current_confused)
            prev_pos = agent.current_pos

            result = env.step(actions) 

            reward = compute_reward(
                result,
                prev_pos,
                env.goal,
                shaping_weight,
            )
            next_state = agent.state_to_key(result.current_position, result.is_confused)
            agent.update_q(state, action, reward, next_state)

            episode_reward += reward
            episode_turns += 1

            if result.is_dead:
                deaths_this_episode += 1

            last_result = result

            if result.is_goal_reached:
                break
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_steps.append(episode_turns)
        episode_successes.append(1 if result.is_goal_reached else 0)
        episode_death_counts.append(deaths_this_episode)
        
        # Log progress
        if (episode + 1) % log_interval == 0:
            window = min(log_interval, len(episode_rewards))
            avg_reward = sum(episode_rewards[-window:]) / window
            avg_steps = sum(episode_steps[-window:]) / window
            success_rate = sum(episode_successes[-window:]) / window * 100
            death_rate = sum(episode_death_counts[-window:]) / window * 100
            
            elapsed = time.time() - start_time
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Steps: {avg_steps:.1f} | "
                  f"Success Rate: {success_rate:.1f}% | "
                  f"Death Rate: {death_rate:.1f}% | "
                f"ShapeW: {shaping_weight:.3f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Time: {elapsed:.1f}s")
        
        # Save checkpoint
        if (episode + 1) % checkpoint_interval == 0:
            checkpoint_path = f"checkpoints/q_agent_ep{episode + 1}.pkl"
            agent.save_q_table(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    agent.save_q_table("q_agent_final.pkl")
    print(f"\nTraining complete! Saved final model to q_agent_final.pkl")
    
    # Print summary
    total_time = time.time() - start_time
    print(f"\nTraining Summary:")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Episodes: {num_episodes}")
    print(f"  Final epsilon: {agent.epsilon:.4f}")
    summary_window = min(log_interval, len(episode_rewards)) or 1
    print(f"  Average final reward: {sum(episode_rewards[-summary_window:]) / summary_window:.2f}")
    print(f"  Final success rate: {sum(episode_successes[-summary_window:]) / summary_window * 100:.1f}%")
    print(f"  Q-table size: {len(agent.q_table)} state-action pairs")


if __name__ == "__main__":
    # Start training
    train_qlearning(
        image_path="maze_5_edited.png",
        num_episodes=250,
        max_steps=5000,
        log_interval=10,
        checkpoint_interval=50,
    )
