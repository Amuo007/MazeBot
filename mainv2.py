from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from agent import MazeAgent
from blind_explore import BlindKnowledge, build_blind_knowledge, shortest_path_in_discovered
from environment import Action, MazeEnvironment, START, GOAL, UNKNOWN, TurnResult
from sarsa import QLearner
from visualizer import animate_episode

Cell = Tuple[int, int]

IMAGE_PATH = "maze_beta.png"
ANIMATION_FRAME_MS = 100

MAX_TURNS_PER_EP = 5000
QTABLE_PATH = "qtable.json"


def discovered_wall_matrices(knowledge: BlindKnowledge) -> Tuple[np.ndarray, np.ndarray]:
    vertical_walls = np.ones((knowledge.n, knowledge.n + 1), dtype=np.uint8)
    horizontal_walls = np.ones((knowledge.n + 1, knowledge.n), dtype=np.uint8)

    for a, b in knowledge.open_edges:
        ar, ac = a
        br, bc = b
        if ar == br and abs(ac - bc) == 1:
            vertical_walls[ar, max(ac, bc)] = 0
        elif ac == bc and abs(ar - br) == 1:
            horizontal_walls[max(ar, br), ac] = 0

    return vertical_walls, horizontal_walls


def discovered_blocked_wall_matrices(knowledge: BlindKnowledge) -> Tuple[np.ndarray, np.ndarray]:
    vertical_walls = np.zeros((knowledge.n, knowledge.n + 1), dtype=np.uint8)
    horizontal_walls = np.zeros((knowledge.n + 1, knowledge.n), dtype=np.uint8)

    for a, b in knowledge.blocked_edges:
        ar, ac = a
        br, bc = b
        if ar == br and abs(ac - bc) == 1:
            vertical_walls[ar, max(ac, bc)] = 1
        elif ac == bc and abs(ar - br) == 1:
            horizontal_walls[max(ar, br), ac] = 1

    return vertical_walls, horizontal_walls


def discovered_obj_matrix(knowledge: BlindKnowledge) -> np.ndarray:
    obj_matrix = np.full((knowledge.n, knowledge.n), UNKNOWN, dtype=np.int32)

    for cell, tile in knowledge.tile_of.items():
        obj_matrix[cell] = tile

    obj_matrix[knowledge.start] = START
    obj_matrix[knowledge.goal] = GOAL
    return obj_matrix


class RouteExecutionAgent(MazeAgent):
    def __init__(
        self,
        start: Cell,
        goal: Cell,
        vertical_walls,
        horizontal_walls,
        obj_matrix,
        teleport_pairs,
        seed_knowledge: BlindKnowledge,
        fixed_route: List[Cell],
        qlearner: Optional[QLearner] = None,
        env=None,
    ):
        self.seed_knowledge = seed_knowledge.clone()
        self.fixed_route = list(fixed_route)
        self.route_index: Dict[Cell, int] = {cell: idx for idx, cell in enumerate(self.fixed_route)}
        self.route_progress_index = 0
        super().__init__(
            start=start,
            goal=goal,
            vertical_walls=vertical_walls,
            horizontal_walls=horizontal_walls,
            obj_matrix=obj_matrix,
            teleport_pairs=teleport_pairs,
            qlearner=qlearner,
            env=env,
        )

    def reset_episode(self) -> None:
        super().reset_episode()
        self.memory.visited.update(self.seed_knowledge.visited)
        self.memory.known_safe.update(self.seed_knowledge.visited)
        self.memory.known_safe.discard(self.goal)
        self.route_progress_index = 0
        self.current_path = list(self.fixed_route)

    def update_from_result(self, result: Optional[TurnResult]) -> None:
        super().update_from_result(result)
        idx = self.route_index.get(self.current_pos)
        if idx is not None:
            self.route_progress_index = max(self.route_progress_index, idx)

    def _candidate_steps(self, cell: Cell) -> List[Tuple[Cell, Cell]]:
        r, c = cell
        candidates: List[Tuple[Cell, Cell]] = []

        for nb in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if not self.can_move(cell, nb):
                continue
            candidates.append((nb, self.teleport_pairs.get(nb, nb)))

        return candidates

    def neighbors(self, cell: Cell) -> List[Cell]:
        landings: List[Cell] = []
        seen: Set[Cell] = set()
        for _step_cell, landing in self._candidate_steps(cell):
            if landing in seen:
                continue
            seen.add(landing)
            landings.append(landing)
        return landings

    def _build_route_suffix(self) -> List[Cell]:
        idx = self.route_index.get(self.current_pos)
        if idx is not None:
            self.route_progress_index = max(self.route_progress_index, idx)
            return self.fixed_route[idx:]

        best_index: Optional[int] = None
        for _step_cell, landing in self._candidate_steps(self.current_pos):
            idx = self.route_index.get(landing)
            if idx is None or idx < self.route_progress_index:
                continue
            if best_index is None or idx < best_index:
                best_index = idx

        if best_index is None:
            return [self.current_pos]

        return [self.current_pos] + self.fixed_route[best_index:]

    def _replan(self) -> None:
        self.current_path = self._build_route_suffix()

    def astar_suggestion(self) -> Optional[Action]:
        if len(self.current_path) < 2:
            return None

        target_landing = self.current_path[1]
        for nb, landing in self._candidate_steps(self.current_pos):
            if landing == target_landing:
                return self.controller.delta_to_action(self.current_pos, nb)

        return None


def build_endgame_agent(
    env: MazeEnvironment,
    knowledge: BlindKnowledge,
    discovered_path: List[Cell],
    qlearner: QLearner,
) -> RouteExecutionAgent:
    vertical_walls, horizontal_walls = discovered_wall_matrices(knowledge)
    obj_matrix = discovered_obj_matrix(knowledge)
    agent = RouteExecutionAgent(
        start=env.start,
        goal=env.goal,
        vertical_walls=vertical_walls,
        horizontal_walls=horizontal_walls,
        obj_matrix=obj_matrix,
        teleport_pairs=knowledge.teleport_pairs,
        seed_knowledge=knowledge,
        fixed_route=discovered_path,
        qlearner=qlearner,
        env=env,
    )
    display_vertical_walls, display_horizontal_walls = discovered_blocked_wall_matrices(knowledge)
    agent.display_obj_matrix = obj_matrix
    agent.display_vertical_walls = display_vertical_walls
    agent.display_horizontal_walls = display_horizontal_walls
    return agent


def main() -> None:
    preview_env = MazeEnvironment(image_path=IMAGE_PATH, maze_size=64)
    print(f"Start : {preview_env.start}", flush=True)
    print(f"Goal  : {preview_env.goal}", flush=True)

    print("\n── Blind exploration phase ──", flush=True)
    knowledge, exploration_episodes = build_blind_knowledge(IMAGE_PATH)
    discovered_path = shortest_path_in_discovered(knowledge)

    print(f"Exploration episodes : {exploration_episodes}", flush=True)
    print(f"Discovered cells     : {len(knowledge.visited)}", flush=True)
    print(f"Discovered walls     : {len(knowledge.blocked_edges) // 2}", flush=True)
    print(f"Discovered teleports : {len(knowledge.teleport_pairs) // 2}", flush=True)
    print(f"Goal seen in map     : {'yes' if knowledge.goal in knowledge.visited else 'no'}", flush=True)

    if not discovered_path:
        raise RuntimeError("Exploration did not produce a discovered route to the goal.")

    print(f"Discovered path len  : {len(discovered_path) - 1}", flush=True)

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
    if not loaded:
        raise FileNotFoundError(
            f"Expected an existing RL Q-table at {QTABLE_PATH}; mainv2 no longer trains automatically."
        )

    print(f"[mainv2] Loaded RL Q-table from {QTABLE_PATH}.", flush=True)

    agent = build_endgame_agent(env, knowledge, discovered_path, qlearner)

    print("\n── RL endgame phase ──", flush=True)
    print("RL now executes the fixed route discovered during exploration.", flush=True)

    qlearner.epsilon = 0.0
    env.reset()
    agent.reset_episode()
    print("\n── Running final visualised endgame episode ──", flush=True)
    animate_episode(env, agent, max_turns=10000, frame_ms=ANIMATION_FRAME_MS)

    print("\nEpisode stats:")
    print(env.get_episode_stats())


if __name__ == "__main__":
    main()
