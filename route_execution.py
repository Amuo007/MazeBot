from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from agent import MazeAgent
from environment import Action, GOAL, START, UNKNOWN, MazeEnvironment, TurnResult
from exploration.knowledge import BlindKnowledge
from qlearning import QLearner

Cell = Tuple[int, int]


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
        self.route_index: Dict[Cell, int] = {cell: index for index, cell in enumerate(self.fixed_route)}
        self.route_progress_index = 0
        super().__init__(
            start=start,
            goal=goal,
            vertical_walls=vertical_walls,
            horizontal_walls=horizontal_walls,
            obj_matrix=obj_matrix,
            teleport_pairs=teleport_pairs,
            one_way_gates=getattr(env, "one_way_gates", None),
            qlearner=qlearner,
            env=env,
        )

    def reset_episode(self) -> None:
        super().reset_episode()
        self.memory.visited.update(self.seed_knowledge.visited)
        self.route_progress_index = 0
        self.current_path = list(self.fixed_route)

    def update_from_result(self, result: Optional[TurnResult]) -> None:
        super().update_from_result(result)
        route_index = self.route_index.get(self.current_pos)
        if route_index is not None:
            self.route_progress_index = max(self.route_progress_index, route_index)

    def _candidate_steps(self, cell: Cell) -> List[Tuple[Cell, Cell]]:
        row, col = cell
        candidates: List[Tuple[Cell, Cell]] = []

        for neighbor in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]:
            if not self.can_move(cell, neighbor):
                continue
            candidates.append((neighbor, self.teleport_pairs.get(neighbor, neighbor)))

        return candidates

    def _build_route_suffix(self) -> List[Cell]:
        route_index = self.route_index.get(self.current_pos)
        if route_index is not None:
            self.route_progress_index = max(self.route_progress_index, route_index)
            return self.fixed_route[route_index:]

        best_index: Optional[int] = None
        for _, landing in self._candidate_steps(self.current_pos):
            route_index = self.route_index.get(landing)
            if route_index is None or route_index < self.route_progress_index:
                continue
            if best_index is None or route_index < best_index:
                best_index = route_index

        if best_index is None:
            return [self.current_pos]

        return [self.current_pos] + self.fixed_route[best_index:]

    def _replan(self) -> None:
        self.current_path = self._build_route_suffix()

    def astar_suggestion(self) -> Optional[Action]:
        if len(self.current_path) < 2:
            return None

        target_landing = self.current_path[1]
        for neighbor, landing in self._candidate_steps(self.current_pos):
            if landing == target_landing:
                return self.controller.delta_to_action(self.current_pos, neighbor)

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
