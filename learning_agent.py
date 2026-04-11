import pickle
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from environment import Action, Cell, TurnResult


@dataclass
class AgentMemory:
    known_walls: Set[Tuple[Cell, Cell]] = field(default_factory=set)
    known_safe: Set[Cell] = field(default_factory=set)
    known_pits: Set[Cell] = field(default_factory=set)
    known_confusion: Set[Cell] = field(default_factory=set)
    known_teleports: Dict[Cell, Cell] = field(default_factory=dict)
    visited: Set[Cell] = field(default_factory=set)


class QLearningAgent:
    """
    Tabular Q-learning agent with persistent memory and turn-based planning.

    The agent conforms more closely to the project spec by exposing a
    plan_turn(last_result) interface and retaining discovered map information
    across episodes.
    """

    def __init__(
        self,
        start: Cell = (0, 0),
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
    ):
        self.start = start
        self.current_pos = start
        self.current_confused = False

        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.98

        self.memory = AgentMemory()
        self.episode_visited: Set[Cell] = set()

        self.last_result: Optional[TurnResult] = None
        self.last_action: Optional[Action] = None
        self.last_position: Cell = start

    def state_to_key(self, pos: Cell, confused: bool) -> tuple:
        """Encode state into a small hashable key."""
        return (pos[0], pos[1], bool(confused))

    def _action_target(self, position: Cell, action: Action) -> Cell:
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

    def observe_result(
        self,
        result: TurnResult,
        action: Optional[Action] = None,
        previous_position: Optional[Cell] = None,
    ) -> None:
        """Update persistent memory from turn feedback."""
        self.last_result = result
        self.current_pos = result.current_position
        self.current_confused = result.is_confused

        self.memory.visited.add(result.current_position)
        self.memory.known_safe.add(result.current_position)
        self.episode_visited.add(result.current_position)

        if result.is_confused:
            self.memory.known_confusion.add(result.current_position)

        if result.is_dead:
            self.memory.known_pits.add(result.current_position)

        if action is not None and previous_position is not None:
            if result.wall_hits > 0 and action != Action.WAIT:
                self.memory.known_walls.add(
                    (previous_position, self._action_target(previous_position, action))
                )

            if result.teleported and previous_position != result.current_position:
                self.memory.known_teleports[previous_position] = result.current_position

    def reset_episode(self, start: Optional[Cell] = None):
        """Reset episode-scoped state while keeping learned memory."""
        if start is not None:
            self.start = start

        self.current_pos = self.start
        self.current_confused = False
        self.last_result = None
        self.last_action = None
        self.last_position = self.start
        self.episode_visited = {self.start}

        self.memory.visited.add(self.start)
        self.memory.known_safe.add(self.start)

    def get_action(self, state: tuple, epsilon: float) -> Action:
        """Select an action using epsilon-greedy exploration."""
        if random.random() < epsilon:
            return random.choice(list(Action))

        q_values = [self.q_table.get((state, action), 0.0) for action in Action]
        max_q = max(q_values)
        best_actions = [
            action
            for action in Action
            if self.q_table.get((state, action), 0.0) == max_q
        ]
        return random.choice(best_actions)

    def get_max_q_next(self, next_state: tuple) -> float:
        """Return the best Q-value for the next state."""
        return max(self.q_table.get((next_state, action), 0.0) for action in Action)

    def update_q(self, state: tuple, action: Action, reward: float, next_state: tuple):
        """Apply the Q-learning update rule."""
        old_q = self.q_table.get((state, action), 0.0)
        max_q_next = self.get_max_q_next(next_state)
        new_q = old_q + self.alpha * (reward + self.gamma * max_q_next - old_q)
        self.q_table[(state, action)] = new_q

    def plan_turn(self, last_result: Optional[TurnResult]) -> List[Action]:
        """Update state from the previous turn and plan the next atomic action."""
        if last_result is not None:
            self.observe_result(
                last_result,
                action=self.last_action,
                previous_position=self.last_position,
            )

        state = self.state_to_key(self.current_pos, self.current_confused)
        action = self.get_action(state, self.epsilon)

        self.last_position = self.current_pos
        self.last_action = action
        return [action]

    def decay_epsilon(self):
        """Decay exploration rate for annealing."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self, filepath: str):
        """Save Q-table to file using pickle."""
        with open(filepath, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filepath: str):
        """Load Q-table from file using pickle."""
        with open(filepath, "rb") as f:
            self.q_table = pickle.load(f)
