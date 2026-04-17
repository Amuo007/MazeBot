from __future__ import annotations

import json
import os
import random
from enum import Enum
from typing import Dict, List, Tuple

Cell = Tuple[int, int]

# state = (row, col, confused_flag, fire_phase)
State = Tuple[int, int, int, int]


class MetaAction(Enum):
    FOLLOW_ASTAR          = 0
    FOLLOW_ASTAR_INVERTED = 1
    WAIT                  = 2


META_ACTIONS = [
    MetaAction.FOLLOW_ASTAR,
    MetaAction.FOLLOW_ASTAR_INVERTED,
    MetaAction.WAIT,
]

# Actions available when NOT confused — INVERTED is locked out
ACTIONS_NORMAL   = [MetaAction.FOLLOW_ASTAR, MetaAction.WAIT]

# Actions available when confused — all three allowed
ACTIONS_CONFUSED = [MetaAction.FOLLOW_ASTAR, MetaAction.FOLLOW_ASTAR_INVERTED, MetaAction.WAIT]


# =========================
# REWARDS
# =========================
REWARD_GOAL           =  100.0
REWARD_DEATH          = -100.0
REWARD_WALL_HIT       =  -10.0
REWARD_FOLLOW_SUCCESS =    2.0   # rewarded for BOTH normal and inverted when they result in movement
REWARD_WAIT           =   -1.0
REWARD_STEP_COST      =   -0.5
REWARD_NO_PROGRESS    =   -2.0


class QLearner:
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: Dict[State, list] = {}

    def _get_q(self, state: State) -> list:
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(META_ACTIONS)
        return self.q_table[state]

    def _available_actions(self, state: State) -> List[MetaAction]:
        """Only allow INVERTED when the agent is actually confused."""
        _, _, confused_flag, _ = state
        return ACTIONS_CONFUSED if confused_flag == 1 else ACTIONS_NORMAL

    def best_action(self, state: State) -> MetaAction:
        available = self._available_actions(state)
        q_vals    = self._get_q(state)
        return max(available, key=lambda a: q_vals[META_ACTIONS.index(a)])

    def max_q_value(self, state: State) -> float:
        available = self._available_actions(state)
        q_vals    = self._get_q(state)
        return max(q_vals[META_ACTIONS.index(a)] for a in available)

    def select_action(self, state: State) -> MetaAction:
        available = self._available_actions(state)
        if random.random() < self.epsilon:
            return random.choice(available)
        return self.best_action(state)

    def update(
        self,
        state:      State,
        action:     MetaAction,
        reward:     float,
        next_state: State,
    ) -> None:
        ai        = META_ACTIONS.index(action)
        q_sa      = self._get_q(state)[ai]
        q_next    = self.max_q_value(next_state)
        td_target = reward + self.gamma * q_next
        td_error  = td_target - q_sa
        self.q_table[state][ai] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # =========================
    # REWARD FUNCTION
    # =========================
    @staticmethod
    def compute_reward(
        is_dead:            bool,
        is_goal:            bool,
        wall_hits:          int,
        chosen_meta_action: MetaAction,
        moved:              bool,
    ) -> float:

        if is_goal:
            return REWARD_GOAL

        if is_dead:
            return REWARD_DEATH

        reward = 0.0

        # Step cost always applied
        reward += REWARD_STEP_COST

        # Wall penalty
        if wall_hits > 0:
            reward += REWARD_WALL_HIT * wall_hits

        # Reward movement for BOTH normal and inverted (inverted is only
        # available when confused, so if it moved, it was the right call)
        if chosen_meta_action in (MetaAction.FOLLOW_ASTAR, MetaAction.FOLLOW_ASTAR_INVERTED) and moved:
            reward += REWARD_FOLLOW_SUCCESS

        # Wait penalty
        if chosen_meta_action == MetaAction.WAIT:
            reward += REWARD_WAIT

        # No progress penalty
        if not moved and wall_hits == 0 and chosen_meta_action != MetaAction.WAIT:
            reward += REWARD_NO_PROGRESS

        return reward

    # =========================
    # SAVE / LOAD
    # =========================
    def save(self, path: str = "qtable.json") -> None:
        serialisable = {str(k): v for k, v in self.q_table.items()}
        with open(path, "w") as f:
            json.dump({"q_table": serialisable, "epsilon": self.epsilon}, f)
        print(f"[Q-learning] Q-table saved → {path} ({len(self.q_table)} states)")

    def load(self, path: str = "qtable.json") -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "r") as f:
            data = json.load(f)
        self.q_table = {eval(k): v for k, v in data["q_table"].items()}
        self.epsilon = data.get("epsilon", self.epsilon_min)
        print(f"[Q-learning] Q-table loaded ← {path} ({len(self.q_table)} states)")
        return True
