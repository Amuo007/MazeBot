from __future__ import annotations

import json
import os
import random
import re
from ast import literal_eval
from enum import Enum
from typing import Dict, List, Tuple

from environment.models import ACTIONS_PER_TURN

State = Tuple[int, int, int, int]


class MetaAction(Enum):
    # Keep values and order stable so existing saved Q-tables stay compatible.
    FOLLOW_PATH = 0
    FOLLOW_PATH_INVERTED = 1
    WAIT = 2


META_ACTIONS = [
    MetaAction.FOLLOW_PATH,
    MetaAction.FOLLOW_PATH_INVERTED,
    MetaAction.WAIT,
]

ACTIONS_NORMAL = [MetaAction.FOLLOW_PATH, MetaAction.WAIT]
ACTIONS_CONFUSED = [MetaAction.FOLLOW_PATH, MetaAction.FOLLOW_PATH_INVERTED, MetaAction.WAIT]
REWARD_GOAL = 100.0
REWARD_DEATH = -100.0
REWARD_WALL_HIT = -10.0
REWARD_FOLLOW_SUCCESS = 2.0
REWARD_WAIT = -2.0
REWARD_STEP_COST = -0.5
REWARD_NO_PROGRESS = -2.0

DEFAULT_QLEARNER_CONFIG = {
    "alpha": 0.1,
    "gamma": 0.95,
    "epsilon": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.995,
}

QTABLE_SCHEMA_VERSION = 2


class QLearner:
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: Dict[State, list] = {}

    @staticmethod
    def _canonical_state(state: State) -> State:
        return tuple(int(value) for value in state)

    @staticmethod
    def _parse_state_key(key: str) -> State:
        try:
            parsed = literal_eval(key)
        except (SyntaxError, ValueError):
            # Older checkpoints may serialize keys like "np.int64(7)".
            key_sanitized = re.sub(r"np\.(?:u?int)\d+\(([-+]?\d+)\)", r"\1", key)
            parsed = literal_eval(key_sanitized)

        if not isinstance(parsed, (tuple, list)):
            raise ValueError(f"Invalid state key format: {key}")

        return tuple(int(value) for value in parsed)

    def _get_q(self, state: State) -> list:
        state = self._canonical_state(state)
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(META_ACTIONS)
        return self.q_table[state]

    def _available_actions(self, state: State) -> List[MetaAction]:
        state = self._canonical_state(state)
        _, _, confused_flag, _ = state
        return ACTIONS_CONFUSED if confused_flag == 1 else ACTIONS_NORMAL

    def best_action(self, state: State) -> MetaAction:
        state = self._canonical_state(state)
        available = self._available_actions(state)
        q_vals = self._get_q(state)
        return max(available, key=lambda action: q_vals[META_ACTIONS.index(action)])

    def max_q_value(self, state: State) -> float:
        state = self._canonical_state(state)
        available = self._available_actions(state)
        q_vals = self._get_q(state)
        return max(q_vals[META_ACTIONS.index(action)] for action in available)

    def select_action(self, state: State) -> MetaAction:
        state = self._canonical_state(state)
        available = self._available_actions(state)
        if random.random() < self.epsilon:
            return random.choice(available)
        return self.best_action(state)

    def update(
        self,
        state: State,
        action: MetaAction,
        reward: float,
        next_state: State,
    ) -> None:
        state = self._canonical_state(state)
        next_state = self._canonical_state(next_state)
        action_index = META_ACTIONS.index(action)
        q_sa = self._get_q(state)[action_index]
        q_next = self.max_q_value(next_state)
        td_target = reward + self.gamma * q_next
        td_error = td_target - q_sa
        self.q_table[state][action_index] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    @staticmethod
    def compute_reward(
        is_dead: bool,
        is_goal: bool,
        wall_hits: int,
        chosen_meta_action: MetaAction,
        moved: bool,
    ) -> float:
        if is_goal:
            return REWARD_GOAL

        if is_dead:
            return REWARD_DEATH

        reward = REWARD_STEP_COST

        if wall_hits > 0:
            reward += REWARD_WALL_HIT * wall_hits

        if chosen_meta_action in (MetaAction.FOLLOW_PATH, MetaAction.FOLLOW_PATH_INVERTED) and moved:
            reward += REWARD_FOLLOW_SUCCESS

        if chosen_meta_action == MetaAction.WAIT:
            reward += REWARD_WAIT

        if not moved and wall_hits == 0 and chosen_meta_action != MetaAction.WAIT:
            reward += REWARD_NO_PROGRESS

        return reward

    def save(self, path: str = "qtable.json") -> None:
        serialisable = {
            str(self._canonical_state(key)): value
            for key, value in self.q_table.items()
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "schema_version": QTABLE_SCHEMA_VERSION,
                    "actions_per_turn": ACTIONS_PER_TURN,
                    "q_table": serialisable,
                    "epsilon": self.epsilon,
                },
                file,
            )
        print(f"[Q-learning] Q-table saved → {path} ({len(self.q_table)} states)")

    def load(self, path: str = "qtable.json") -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        schema_version = data.get("schema_version")
        actions_per_turn = data.get("actions_per_turn")
        if schema_version != QTABLE_SCHEMA_VERSION or actions_per_turn != ACTIONS_PER_TURN:
            print(
                "[Q-learning] Existing Q-table is from an incompatible turn model; "
                "please retrain to regenerate qtable.json."
            )
            return False

        self.q_table = {
            self._parse_state_key(key): value
            for key, value in data["q_table"].items()
        }
        self.epsilon = data.get("epsilon", self.epsilon_min)
        print(f"[Q-learning] Q-table loaded ← {path} ({len(self.q_table)} states)")
        return True


def build_default_qlearner(**overrides: float) -> QLearner:
    config = {**DEFAULT_QLEARNER_CONFIG, **overrides}
    return QLearner(**config)
