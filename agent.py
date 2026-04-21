from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from astar import astar_search
from environment import ACTIONS_PER_TURN, Action, TurnResult
from qlearning import MetaAction, QLearner, State

Cell = Tuple[int, int]


@dataclass
class AgentMemory:
    visited: Set[Cell] = field(default_factory=set)


class ActionController:
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

    @staticmethod
    def invert_action(action: Action) -> Action:
        if action == Action.MOVE_UP:
            return Action.MOVE_DOWN
        if action == Action.MOVE_DOWN:
            return Action.MOVE_UP
        if action == Action.MOVE_LEFT:
            return Action.MOVE_RIGHT
        if action == Action.MOVE_RIGHT:
            return Action.MOVE_LEFT
        return Action.WAIT


class MazeAgent:
    def __init__(
        self,
        start: Cell,
        goal: Cell,
        vertical_walls,
        horizontal_walls,
        obj_matrix,
        teleport_pairs,
        one_way_gates: Optional[dict] = None,
        qlearner: Optional[QLearner] = None,
        env=None,
    ):
        self.start = start
        self.goal = goal
        self.vertical_walls = vertical_walls
        self.horizontal_walls = horizontal_walls
        self.obj_matrix = obj_matrix
        self.teleport_pairs = teleport_pairs
        self.one_way_gates = dict(one_way_gates) if one_way_gates is not None else {}
        self.env = env

        self.rows, self.cols = obj_matrix.shape
        self.controller = ActionController()
        self.memory = AgentMemory()
        self.qlearner: QLearner = qlearner if qlearner is not None else QLearner()

        self.current_pos: Cell = start
        self.current_path: List[Cell] = []
        self.last_result: Optional[TurnResult] = None

        self._last_state: Optional[State] = None
        self._last_meta_action: Optional[MetaAction] = None
        self._last_pos_before_action: Optional[Cell] = None
        self.confused_turns_remaining = 0

    def reset_episode(self) -> None:
        self.current_pos = self.start
        self.current_path = []
        self.last_result = None
        self._last_state = None
        self._last_meta_action = None
        self._last_pos_before_action = None
        self.confused_turns_remaining = 0

        self.memory.visited.clear()
        self.memory.visited.add(self.start)

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def can_move(self, a: Cell, b: Cell) -> bool:
        ar, ac = a
        br, bc = b
        if not self.in_bounds(b):
            return False

        gate_exit = self.one_way_gates.get(a)
        if gate_exit is not None and b != gate_exit:
            return False

        target_gate_exit = self.one_way_gates.get(b)
        if target_gate_exit is not None and target_gate_exit == a:
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
        return [
            nb for nb in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            if self.can_move(cell, nb)
        ]

    def update_from_result(self, result: Optional[TurnResult]) -> None:
        if result is None:
            return

        self.last_result = result
        self.current_pos = result.current_position
        self.memory.visited.add(self.current_pos)

        if result.is_confused:
            self.confused_turns_remaining = 2

    def _needs_replan(self, last_result: Optional[TurnResult]) -> bool:
        if not self.current_path:
            return True
        if last_result is None:
            return True
        if last_result.is_dead:
            return True
        if last_result.teleported:
            return True
        if self.current_pos not in self.current_path:
            return True
        return False

    def _replan(self) -> None:
        path = astar_search(self.current_pos, self.goal, self.neighbors)
        self.current_path = path if path else [self.current_pos]

    def _advance_path(self) -> None:
        if self.current_pos in self.current_path:
            idx = self.current_path.index(self.current_pos)
            self.current_path = self.current_path[idx:]

    def path_suggestion(self) -> Optional[Action]:
        if len(self.current_path) < 2:
            return None
        return self.controller.delta_to_action(self.current_path[0], self.current_path[1])

    def get_state(self) -> State:
        r, c = self.current_pos
        fire_phase = self.env.get_fire_phase() if self.env is not None else 0
        confused_flag = 1 if self.confused_turns_remaining > 0 else 0
        return (r, c, confused_flag, fire_phase)

    def choose_primitive_action(self, meta_action: MetaAction) -> Action:
        if meta_action == MetaAction.WAIT:
            return Action.WAIT

        suggestion = self.path_suggestion()
        base_action = suggestion if suggestion is not None else Action.WAIT

        if meta_action == MetaAction.FOLLOW_PATH_INVERTED:
            return self.controller.invert_action(base_action)

        return base_action

    @staticmethod
    def _action_to_target(position: Cell, action: Action) -> Cell:
        row, col = position
        if action == Action.MOVE_UP:
            return (row - 1, col)
        if action == Action.MOVE_DOWN:
            return (row + 1, col)
        if action == Action.MOVE_LEFT:
            return (row, col - 1)
        if action == Action.MOVE_RIGHT:
            return (row, col + 1)
        return position

    def _simulate_transition(self, position: Cell, action: Action) -> Cell:
        if action == Action.WAIT:
            return position

        target = self._action_to_target(position, action)
        if not self.can_move(position, target):
            return position
        return self.teleport_pairs.get(target, target)

    def _build_follow_path_actions(self, steps: int = ACTIONS_PER_TURN) -> List[Action]:
        actions: List[Action] = []
        saved_pos = self.current_pos
        saved_path = list(self.current_path)

        try:
            for _ in range(steps):
                if self.current_pos == self.goal:
                    actions.append(Action.WAIT)
                    continue

                if not self.current_path or self.current_pos not in self.current_path:
                    self._replan()
                else:
                    self._advance_path()

                suggestion = self.path_suggestion()
                next_action = suggestion if suggestion is not None else Action.WAIT
                actions.append(next_action)
                self.current_pos = self._simulate_transition(self.current_pos, next_action)

            return actions
        finally:
            self.current_pos = saved_pos
            self.current_path = saved_path

    def plan_turn(self, last_result: Optional[TurnResult]) -> List[Action]:
        self.update_from_result(last_result)

        if self.current_pos == self.goal:
            return [Action.WAIT] * ACTIONS_PER_TURN

        current_state = self.get_state()

        if self._needs_replan(last_result):
            self._replan()
        else:
            self._advance_path()

        if (
            self._last_state is not None
            and self._last_meta_action is not None
            and last_result is not None
            and self._last_pos_before_action is not None
        ):
            moved = (self.current_pos != self._last_pos_before_action)

            reward = QLearner.compute_reward(
                is_dead=last_result.is_dead,
                is_goal=last_result.is_goal_reached,
                wall_hits=last_result.wall_hits,
                chosen_meta_action=self._last_meta_action,
                moved=moved,
            )

            self.qlearner.update(
                self._last_state,
                self._last_meta_action,
                reward,
                current_state,
            )

        chosen_meta_action = self.qlearner.select_action(current_state)
        follow_actions = self._build_follow_path_actions(ACTIONS_PER_TURN)
        if chosen_meta_action == MetaAction.WAIT:
            turn_actions = [Action.WAIT] * ACTIONS_PER_TURN
        elif chosen_meta_action == MetaAction.FOLLOW_PATH_INVERTED:
            turn_actions = [self.controller.invert_action(action) for action in follow_actions]
        else:
            turn_actions = follow_actions

        self._last_state = current_state
        self._last_meta_action = chosen_meta_action
        self._last_pos_before_action = self.current_pos

        if self.confused_turns_remaining > 0:
            self.confused_turns_remaining -= 1

        return turn_actions
