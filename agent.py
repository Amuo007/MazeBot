from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from astar import astar_search_debug
from environment import Action, TurnResult

Cell = Tuple[int, int]


@dataclass
class AgentMemory:
    known_walls: Set[Tuple[Cell, Cell]] = field(default_factory=set)
    known_safe: Set[Cell] = field(default_factory=set)
    known_pits: Set[Cell] = field(default_factory=set)
    known_confusion: Set[Cell] = field(default_factory=set)
    known_teleports: Dict[Cell, Cell] = field(default_factory=dict)
    visited: Set[Cell] = field(default_factory=set)


class ActionController:
    @staticmethod
    def move_up() -> Action:
        return Action.MOVE_UP

    @staticmethod
    def move_down() -> Action:
        return Action.MOVE_DOWN

    @staticmethod
    def move_left() -> Action:
        return Action.MOVE_LEFT

    @staticmethod
    def move_right() -> Action:
        return Action.MOVE_RIGHT

    @staticmethod
    def wait() -> Action:
        return Action.WAIT

    @staticmethod
    def delta_to_action(a: Cell, b: Cell) -> Action:
        dr = b[0] - a[0]
        dc = b[1] - a[1]
        if dr == -1 and dc == 0:
            return Action.MOVE_UP
        if dr == 1 and dc == 0:
            return Action.MOVE_DOWN
        if dr == 0 and dc == -1:
            return Action.MOVE_LEFT
        if dr == 0 and dc == 1:
            return Action.MOVE_RIGHT
        return Action.WAIT


class MazeAgent:
    """
    Starter refactor agent.
    Still uses full maze knowledge for dev/testing.
    """

    def __init__(
        self,
        start: Cell,
        goal: Cell,
        vertical_walls,
        horizontal_walls,
        obj_matrix,
        teleport_pairs,
    ):
        self.start = start
        self.goal = goal

        self.vertical_walls = vertical_walls
        self.horizontal_walls = horizontal_walls
        self.obj_matrix = obj_matrix
        self.teleport_pairs = teleport_pairs

        self.rows, self.cols = obj_matrix.shape
        self.controller = ActionController()
        self.memory = AgentMemory()

        self.current_pos: Cell = start
        self.current_path: List[Cell] = []
        self.last_result: Optional[TurnResult] = None

        # for visualization
        self.last_search_expanded: List[Cell] = []
        self.last_search_closed: Set[Cell] = set()

    def reset_episode(self):
        self.current_pos = self.start
        self.current_path = []
        self.last_result = None
        self.last_search_expanded = []
        self.last_search_closed = set()

        self.memory.visited.clear()
        self.memory.known_safe.clear()

        self.memory.visited.add(self.start)
        self.memory.known_safe.add(self.start)

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def can_move(self, a: Cell, b: Cell) -> bool:
        ar, ac = a
        br, bc = b

        if not self.in_bounds(b):
            return False

        if br == ar - 1 and bc == ac:
            return self.horizontal_walls[ar, ac] == 0
        if br == ar + 1 and bc == ac:
            return self.horizontal_walls[ar + 1, ac] == 0
        if br == ar and bc == ac - 1:
            return self.vertical_walls[ar, ac] == 0
        if br == ar and bc == ac + 1:
            return self.vertical_walls[ar, ac + 1] == 0

        return False

    def neighbors(self, cell: Cell) -> List[Cell]:
        r, c = cell
        out = []
        for nb in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if self.can_move(cell, nb):
                out.append(nb)
        return out

    def plan_path(self, start: Cell, goal: Cell) -> List[Cell]:
        debug = astar_search_debug(start, goal, self.neighbors)
        self.last_search_expanded = debug["expanded_order"]
        self.last_search_closed = debug["closed_set"]
        return debug["path"]

    def path_to_actions(self, path: List[Cell], limit: int = 5) -> List[Action]:
        if len(path) < 2:
            return [self.controller.wait()]

        actions: List[Action] = []
        for i in range(len(path) - 1):
            actions.append(self.controller.delta_to_action(path[i], path[i + 1]))
            if len(actions) >= limit:
                break

        return actions if actions else [self.controller.wait()]

    def update_from_result(self, result: Optional[TurnResult]):
        if result is None:
            return

        self.last_result = result
        self.current_pos = result.current_position
        self.memory.visited.add(self.current_pos)
        self.memory.known_safe.add(self.current_pos)

    def plan_turn(self, last_result: Optional[TurnResult]) -> List[Action]:
        self.update_from_result(last_result)

        if self.current_pos == self.goal:
            return [self.controller.wait()]

        self.current_path = self.plan_path(self.current_pos, self.goal)
        if not self.current_path:
            self.current_path = [self.current_pos]
            return [self.controller.wait()]

        return self.path_to_actions(self.current_path, limit=5)