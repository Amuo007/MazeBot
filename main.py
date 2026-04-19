import sys
from agent import MazeAgent
from environment import MazeEnvironment
from sarsa import QLearner
from visualizer import animate_episode

IMAGE_PATH = "maze_11.png"
ANIMATION_FRAME_MS = 100

TRAIN_EPISODES = 500
MAX_TURNS_PER_EP = 5000
QTABLE_PATH = "qtable.json"
VISUALIZE_EVERY = 50


def run_visual_episode(env: MazeEnvironment, agent: MazeAgent, ep: int) -> None:
    """Pause training and show one animated episode."""
    print(f"\n  ── Visual check at episode {ep} ──", flush=True)
    saved_epsilon = agent.qlearner.epsilon
    agent.qlearner.epsilon = 0.0

    env.reset()
    agent.reset_episode()
    animate_episode(env, agent, max_turns=5000, frame_ms=ANIMATION_FRAME_MS)

    agent.qlearner.epsilon = saved_epsilon
    print(f"  ── Resuming training ──\n", flush=True)


def train(env: MazeEnvironment, agent: MazeAgent, episodes: int) -> None:
    print(f"\n── Training for {episodes} episodes ──", flush=True)

    for ep in range(1, episodes + 1):
        env.reset()
        agent.reset_episode()

        last_result = None
        turns = 0

        for _turn in range(MAX_TURNS_PER_EP):
            actions = agent.plan_turn(last_result)
            last_result = env.step(actions)
            turns += 1

            if last_result.is_goal_reached:
                break

        agent.qlearner.decay_epsilon()

        stats = env.get_episode_stats()
        print(
            f"  Ep {ep:>4}/{episodes} | "
            f"turns={turns:>5} | "
            f"deaths={stats['deaths']:>3} | "
            f"actions={stats['total_actions_executed']:>6} | "
            f"goal={'✓' if stats['goal_reached'] else '✗'} | "
            f"ε={agent.qlearner.epsilon:.3f} | "
            f"Q-states={len(agent.qlearner.q_table):>6}",
            flush=True,
        )

        if ep % VISUALIZE_EVERY == 0:
            run_visual_episode(env, agent, ep)

    print("\n── Training complete ──", flush=True)
    agent.qlearner.save(QTABLE_PATH)
    print(f"── Q-table saved to {QTABLE_PATH} ──\n", flush=True)


def main():
    print("Loading environment...", flush=True)
    env = MazeEnvironment(
        image_path=IMAGE_PATH,
        maze_size=64,
    )

    qlearner = QLearner(
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
    )
    loaded = qlearner.load(QTABLE_PATH)
    if loaded:
        print("[main] Loaded existing Q-table — skipping training.", flush=True)

    agent = MazeAgent(
        start=env.start,
        goal=env.goal,
        vertical_walls=env.vertical_walls,
        horizontal_walls=env.horizontal_walls,
        obj_matrix=env.obj_matrix,
        teleport_pairs=env.teleport_pairs,
        qlearner=qlearner,
        env=env,
    )

    print(f"Start : {env.start}", flush=True)
    print(f"Goal  : {env.goal}", flush=True)

    if not loaded:
        train(env, agent, episodes=TRAIN_EPISODES)

    qlearner.epsilon = 0.0
    env.reset()
    agent.reset_episode()
    print("── Running final visualised episode ──", flush=True)
    animate_episode(env, agent, max_turns=10000, frame_ms=ANIMATION_FRAME_MS)

    print("\nEpisode stats:")
    print(env.get_episode_stats())


if __name__ == "__main__":
    main()