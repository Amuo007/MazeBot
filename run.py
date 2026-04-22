from __future__ import annotations

from typing import List, Tuple

from environment import MazeEnvironment
from evaluation import DEFAULT_PHASE2_REPORT_PATH, write_phase2_report
from exploration import BlindKnowledge, Cell, MAZE_SIZE, build_blind_knowledge, shortest_path_in_discovered
from qlearning import QLearner, build_default_qlearner
from route_execution import build_endgame_agent
from visualizer import animate_episode

IMAGE_PATH = "maze_beta.png"
ANIMATION_FRAME_MS = 100
QTABLE_PATH = "qtable.json"


def preview_maze(image_path: str) -> None:
    preview_env = MazeEnvironment(image_path=image_path, maze_size=MAZE_SIZE)
    print(f"Start : {preview_env.start}", flush=True)
    print(f"Goal  : {preview_env.goal}", flush=True)


def run_exploration_phase(image_path: str) -> Tuple[BlindKnowledge, List[Cell], int]:
    print("\n── Blind exploration phase ──", flush=True)
    knowledge, exploration_episodes = build_blind_knowledge(image_path)
    discovered_path = shortest_path_in_discovered(knowledge)

    print(f"Exploration episodes : {exploration_episodes}", flush=True)
    print(f"Discovered cells     : {len(knowledge.visited)}", flush=True)
    print(f"Discovered walls     : {len(knowledge.blocked_edges) // 2}", flush=True)
    print(f"Discovered teleports : {len(knowledge.teleport_pairs) // 2}", flush=True)
    print(f"Goal seen in map     : {'yes' if knowledge.goal in knowledge.visited else 'no'}", flush=True)

    if not discovered_path:
        raise RuntimeError("Exploration did not produce a discovered route to the goal.")

    print(f"Discovered path len  : {len(discovered_path) - 1}", flush=True)
    return knowledge, discovered_path, exploration_episodes


def load_qlearner(path: str) -> QLearner:
    qlearner = build_default_qlearner()
    if not qlearner.load(path):
        raise FileNotFoundError(
            f"Expected a compatible RL Q-table at {path}; run.py does not train it. "
            "Run run_RL.py to retrain and regenerate qtable.json."
        )
    print(f"[run.py] Loaded RL Q-table from {path}.", flush=True)
    return qlearner


def run_endgame_phase(
    image_path: str,
    knowledge: BlindKnowledge,
    discovered_path: List[Cell],
    qlearner: QLearner,
    exploration_episodes: int,
) -> None:
    print("\n── RL endgame phase ──", flush=True)
    print("RL now executes the fixed route discovered during exploration.", flush=True)

    qlearner.epsilon = 0.0
    env = MazeEnvironment(image_path=image_path, maze_size=MAZE_SIZE)
    agent = build_endgame_agent(env, knowledge, discovered_path, qlearner)
    env.reset()
    agent.reset_episode()
    print("\n── Running single visualised endgame episode ──", flush=True)
    run_summary = animate_episode(env, agent, max_turns=10000, frame_ms=ANIMATION_FRAME_MS)

    report = write_phase2_report(
        env=env,
        agent=agent,
        knowledge=knowledge,
        discovered_path=discovered_path,
        run_summary=run_summary,
        report_path=DEFAULT_PHASE2_REPORT_PATH,
        max_turns=10000,
        phase1_exploration_episodes=exploration_episodes,
        qtable_state_count=len(qlearner.q_table),
    )
    print(f"\nPhase 2 report saved to {DEFAULT_PHASE2_REPORT_PATH}.", flush=True)
    print(f"Phase 2 success      : {'yes' if report['success'] else 'no'}", flush=True)
    print(f"Phase 2 turns        : {report['raw_counts']['turns_taken']}", flush=True)
    print(f"Phase 2 actions      : {report['raw_counts']['total_actions_executed']}", flush=True)
    print(f"Phase 2 deaths       : {report['raw_counts']['deaths']}", flush=True)

    print("\nEpisode stats:")
    print(env.get_episode_stats())


def main() -> None:
    preview_maze(IMAGE_PATH)
    knowledge, discovered_path, exploration_episodes = run_exploration_phase(IMAGE_PATH)
    qlearner = load_qlearner(QTABLE_PATH)
    run_endgame_phase(IMAGE_PATH, knowledge, discovered_path, qlearner, exploration_episodes)


if __name__ == "__main__":
    main()
