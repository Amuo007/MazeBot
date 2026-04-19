from __future__ import annotations

import ast
import heapq
import json
import os
import random
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from environment import (
    Action,
    CONFUSION,
    EMPTY,
    FIRE,
    GOAL,
    MazeEnvironment,
    START,
    TP_GREEN,
    TP_PURPLE,
    TP_RED,
    UNKNOWN,
)

Cell = Tuple[int, int]
State = Tuple[int, int, int, int, int, int]

# ============================================================
# CONFIG
# ============================================================
IMAGE_PATH = "maze_6.png"
MAZE_SIZE = 64
MAX_PHYSICAL_STEPS = 20000
FIRE_PHASE_TICKS = 5

TRAIN_EPISODES = 220
TRAINING_STEP_BUDGET = 4500
TRAIN_PRINT_EVERY = 10
SUCCESS_STREAK_TO_STOP = 3
MAX_WAIT_CHAIN = 8
STALL_FRONTIER_TRIGGER = 300

# animation
FRAME_MS = 15
SNAPSHOTS_PER_FRAME = 3
PRINT_EVERY = 250


# ============================================================
# DISPLAY COLORS
# ============================================================
DISPLAY_COLORS = {
    EMPTY: np.array([1.00, 1.00, 1.00]),
    FIRE: np.array([255, 145, 76]) / 255.0,
    CONFUSION: np.array([255, 222, 89]) / 255.0,
    TP_PURPLE: np.array([140, 82, 255]) / 255.0,
    TP_RED: np.array([255, 49, 50]) / 255.0,
    TP_GREEN: np.array([1, 191, 99]) / 255.0,
    START: np.array([15, 192, 223]) / 255.0,
    GOAL: np.array([0, 74, 173]) / 255.0,
    UNKNOWN: np.array([0.68, 0.68, 0.68]),
}
COL_VISITED = np.array([0.60, 0.82, 1.00])
COL_TARGET = np.array([1.00, 0.70, 0.10])
COL_AGENT = np.array([1.00, 0.10, 0.10])
COL_DANGER = np.array([0.45, 0.10, 0.10])
COL_PATH = np.array([1.00, 0.55, 0.15])


# ============================================================
# SMALL UTILS
# ============================================================
def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def adjacent(a: Cell, b: Cell) -> bool:
    return manhattan(a, b) == 1


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


def qtable_path_for(image_path: str) -> str:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return f"blind_qtable_{stem}_stepdir_v9.json"


def fire_cycle_len(env: MazeEnvironment) -> int:
    return max(1, len(env.fire_phase_sets) * FIRE_PHASE_TICKS)


def fire_time_mod(env: MazeEnvironment) -> int:
    return env.total_actions_executed % fire_cycle_len(env)


def fire_phase_for_time_mod(time_mod: int) -> int:
    return time_mod // FIRE_PHASE_TICKS


# ============================================================
# BLIND KNOWLEDGE
#   Only holds what the agent has personally observed.
#   No ground-truth reads from env.vertical_walls / obj_matrix
#   ever happen here for planning.
# ============================================================
class BlindKnowledge:
    def __init__(self, maze_size: int, start: Cell, goal: Cell):
        self.n = maze_size
        self.start = start
        self.goal = goal

        self.visited: Set[Cell] = {start}
        self.tile_of: Dict[Cell, int] = {start: START}

        # edge knowledge (stored bidirectionally)
        self.open_edges: Set[Tuple[Cell, Cell]] = set()
        self.blocked_edges: Set[Tuple[Cell, Cell]] = set()

        # teleport pairs we've actually been bounced through
        self.teleport_pairs: Dict[Cell, Cell] = {}

        # cells we've personally seen burn; visual memory only
        self.dangerous: Set[Cell] = set()
        self.deadly_phases: Dict[Cell, Set[int]] = {}

        # probes given up on this episode due to desyncs
        self.abandoned: Set[Cell] = set()

        # confusion cells seen
        self.confusion_cells: Set[Cell] = set()

    def start_new_episode(self) -> None:
        self.abandoned.clear()
        self.visited.add(self.start)
        self.tile_of[self.start] = START

    def clone(self) -> "BlindKnowledge":
        other = BlindKnowledge(self.n, self.start, self.goal)
        other.visited = set(self.visited)
        other.tile_of = dict(self.tile_of)
        other.open_edges = set(self.open_edges)
        other.blocked_edges = set(self.blocked_edges)
        other.teleport_pairs = dict(self.teleport_pairs)
        other.dangerous = set(self.dangerous)
        other.deadly_phases = {cell: set(phases) for cell, phases in self.deadly_phases.items()}
        other.abandoned = set(self.abandoned)
        other.confusion_cells = set(self.confusion_cells)
        return other

    # ---- geometry ----
    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.n and 0 <= c < self.n

    def candidate_neighbors(self, cell: Cell) -> List[Cell]:
        r, c = cell
        return [nb for nb in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)] if self.in_bounds(nb)]

    def open_neighbors(self, cell: Cell) -> List[Cell]:
        return [nb for nb in self.candidate_neighbors(cell) if (cell, nb) in self.open_edges]

    def planning_neighbors(self, cell: Cell) -> List[Cell]:
        """
        Walking from `cell` into adjacent known-open neighbour `nb` lands on
        `nb`, unless `nb` is a known teleporter, in which case the true landing
        is its paired cell.
        """
        out = []
        for nb in self.open_neighbors(cell):
            out.append(self.teleport_pairs.get(nb, nb))
        return out

    def optimistic_neighbors(self, cell: Cell) -> List[Cell]:
        """
        Goal-directed blind planning graph.
        Unknown edges are treated as traversable until we physically prove
        otherwise by bumping into a wall.
        """
        out = []
        seen = set()
        for nb in self.candidate_neighbors(cell):
            if (cell, nb) in self.blocked_edges:
                continue
            landing = self.teleport_pairs.get(nb, nb)
            if landing in seen:
                continue
            seen.add(landing)
            out.append(landing)
        return out

    def find_walk_step(self, cur: Cell, target_landing: Cell, optimistic: bool = False) -> Optional[Cell]:
        """Which adjacent cell do I step onto to end up at target_landing?"""
        neighbors = self.candidate_neighbors(cur) if optimistic else self.open_neighbors(cur)
        for nb in neighbors:
            if optimistic and (cur, nb) in self.blocked_edges:
                continue
            if self.teleport_pairs.get(nb, nb) == target_landing:
                return nb
        return None

    # ---- edge updates ----
    def mark_open(self, a: Cell, b: Cell) -> bool:
        was_new = (a, b) not in self.open_edges
        self.open_edges.add((a, b))
        self.open_edges.add((b, a))
        return was_new

    def mark_blocked(self, a: Cell, b: Cell) -> bool:
        was_new = (a, b) not in self.blocked_edges
        self.blocked_edges.add((a, b))
        self.blocked_edges.add((b, a))
        return was_new

    def mark_deadly_phase(self, cell: Cell, phase: int) -> bool:
        phases = self.deadly_phases.setdefault(cell, set())
        if phase in phases:
            return False
        phases.add(phase)
        return True

    def is_deadly_at_phase(self, cell: Cell, phase: int) -> bool:
        return phase in self.deadly_phases.get(cell, set())

    def is_deadly_at_time(self, cell: Cell, time_mod: int) -> bool:
        return self.is_deadly_at_phase(cell, fire_phase_for_time_mod(time_mod))

    # ---- frontier ----
    def is_probe_candidate(self, v: Cell, nb: Cell) -> bool:
        if nb in self.visited:
            return False
        if (v, nb) in self.blocked_edges:
            return False
        if nb in self.abandoned:
            return False
        return True

    def reachable_from(self, start: Cell) -> Set[Cell]:
        seen = {start}
        stack = [start]
        while stack:
            cur = stack.pop()
            for nb in self.planning_neighbors(cur):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return seen

    def distances_from(self, start: Cell) -> Dict[Cell, int]:
        dist = {start: 0}
        q = deque([start])
        while q:
            cur = q.popleft()
            for nb in self.planning_neighbors(cur):
                if nb in dist:
                    continue
                dist[nb] = dist[cur] + 1
                q.append(nb)
        return dist

    def pick_frontier_probe(self, current: Cell) -> Optional[Tuple[Cell, Cell]]:
        """
        Greedy frontier pick scored by:
            travel-to-frontier + heuristic-to-goal
        """
        distances = self.distances_from(current)
        best = None
        best_key = None
        for v in self.visited:
            if v not in distances:
                continue
            for nb in self.candidate_neighbors(v):
                if not self.is_probe_candidate(v, nb):
                    continue
                goal_h = manhattan(nb, self.goal)
                f_score = distances[v] + goal_h
                key = (f_score, goal_h, distances[v])
                if best_key is None or key < best_key:
                    best_key = key
                    best = (v, nb)
        return best


# ============================================================
# A* OVER KNOWN GRAPH ONLY
# ============================================================
def plan_known_route(knowledge: BlindKnowledge, start: Cell, target: Cell) -> Optional[List[Cell]]:
    if start == target:
        return [start]

    open_heap: List[Tuple[int, int, int, Cell]] = []
    heapq.heappush(open_heap, (manhattan(start, target), 0, 0, start))

    came_from: Dict[Cell, Cell] = {}
    g_score: Dict[Cell, int] = {start: 0}
    closed: Set[Cell] = set()
    counter = 0

    while open_heap:
        _, g_score_cur, _, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)

        if cur == target:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path

        for nb in knowledge.planning_neighbors(cur):
            tentative = g_score_cur + 1
            if tentative < g_score.get(nb, 10**9):
                g_score[nb] = tentative
                came_from[nb] = cur
                counter += 1
                heapq.heappush(open_heap, (tentative + manhattan(nb, target), tentative, counter, nb))

    return None


def plan_optimistic_route(knowledge: BlindKnowledge, start: Cell, target: Cell) -> Optional[List[Cell]]:
    if start == target:
        return [start]

    open_heap: List[Tuple[int, int, int, Cell]] = []
    heapq.heappush(open_heap, (manhattan(start, target), 0, 0, start))

    came_from: Dict[Cell, Cell] = {}
    g_score: Dict[Cell, int] = {start: 0}
    closed: Set[Cell] = set()
    counter = 0

    while open_heap:
        _, g_score_cur, _, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)

        if cur == target:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path

        for nb in knowledge.optimistic_neighbors(cur):
            tentative = g_score_cur + 1
            if tentative < g_score.get(nb, 10**9):
                g_score[nb] = tentative
                came_from[nb] = cur
                counter += 1
                heapq.heappush(open_heap, (tentative + manhattan(nb, target), tentative, counter, nb))

    return None


def shortest_path_in_discovered(knowledge: BlindKnowledge) -> List[Cell]:
    q = deque([knowledge.start])
    came_from: Dict[Cell, Cell] = {knowledge.start: knowledge.start}

    while q:
        cur = q.popleft()
        if cur == knowledge.goal:
            path = [cur]
            while came_from[cur] != cur:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path

        for nb in knowledge.planning_neighbors(cur):
            if nb in came_from:
                continue
            came_from[nb] = cur
            q.append(nb)

    return []


def plan_timed_route(
    knowledge: BlindKnowledge,
    start: Cell,
    start_time_mod: int,
    target: Cell,
    optimistic: bool,
) -> Optional[List[PlannedStep]]:
    """
    Plan over (cell, fire-clock) states.

    The agent may either:
    - WAIT in place for one action, or
    - step onto an adjacent cell (with known teleport landing if applicable)

    A successor is only allowed if the landing cell is not known-deadly at the
    next fire-clock time.
    """
    if start == target:
        return []

    cycle_len = FIRE_PHASE_TICKS * 4
    start_state = (start, start_time_mod % cycle_len)

    open_heap: List[Tuple[int, int, int, Tuple[Cell, int]]] = []
    heapq.heappush(open_heap, (manhattan(start, target), 0, 0, start_state))

    came_from: Dict[Tuple[Cell, int], Tuple[Tuple[Cell, int], PlannedStep]] = {}
    g_score: Dict[Tuple[Cell, int], int] = {start_state: 0}
    closed: Set[Tuple[Cell, int]] = set()
    counter = 0

    while open_heap:
        _, g_cur, _, state = heapq.heappop(open_heap)
        if state in closed:
            continue
        closed.add(state)

        cell, time_mod = state
        if cell == target:
            plan: List[PlannedStep] = []
            cur = state
            while cur in came_from:
                prev, step = came_from[cur]
                plan.append(step)
                cur = prev
            plan.reverse()
            return plan

        next_time_mod = (time_mod + 1) % cycle_len
        transitions: List[PlannedStep] = []

        if not knowledge.is_deadly_at_time(cell, next_time_mod):
            transitions.append(PlannedStep(step_cell=None, landing=cell))

        neighbors = knowledge.candidate_neighbors(cell) if optimistic else knowledge.open_neighbors(cell)
        for step_cell in neighbors:
            if optimistic and (cell, step_cell) in knowledge.blocked_edges:
                continue
            landing = knowledge.teleport_pairs.get(step_cell, step_cell)
            if knowledge.is_deadly_at_time(landing, next_time_mod):
                continue
            transitions.append(PlannedStep(step_cell=step_cell, landing=landing))

        for step in transitions:
            next_state = (step.landing, next_time_mod)
            tentative = g_cur + 1
            if tentative >= g_score.get(next_state, 10**9):
                continue
            g_score[next_state] = tentative
            came_from[next_state] = (state, step)
            counter += 1
            wait_penalty = 1 if step.step_cell is None else 0
            heapq.heappush(
                open_heap,
                (tentative + manhattan(step.landing, target) + wait_penalty, tentative, counter, next_state),
            )

    return None


# ============================================================
# STEP-LEVEL Q-LEARNING
#   GO = follow the currently planned blind route/probe
#   WAIT = spend one action to shift fire phase
# ============================================================
class StepDecision(Enum):
    GO = 0
    WAIT = 1


STEP_DECISIONS = [StepDecision.GO, StepDecision.WAIT]

REWARD_GOAL = 100.0
REWARD_DEATH = -80.0
REWARD_STEP_COST = -0.20
REWARD_GO_MOVE = 1.80
REWARD_NEW_CELL = 2.50
REWARD_NEW_INFO = 0.60
REWARD_WAIT = -0.70
REWARD_WALL = -0.10
REWARD_DIVERGE = -0.60


@dataclass
class StepOutcome:
    event: str
    moved: bool
    is_dead: bool
    is_goal: bool
    wall_hit: bool
    discovered_new_cell: bool
    discovered_new_info: bool
    teleported: bool


@dataclass(frozen=True)
class PlannedStep:
    step_cell: Optional[Cell]
    landing: Cell


class BlindStepQLearner:
    def __init__(
        self,
        alpha: float = 0.18,
        gamma: float = 0.96,
        epsilon: float = 1.0,
        epsilon_min: float = 0.03,
        epsilon_decay: float = 0.985,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: Dict[State, List[float]] = {}

    def _get_q(self, state: State) -> List[float]:
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(STEP_DECISIONS)
        return self.q_table[state]

    def state_for(self, cell: Cell, planned_step: Cell, env: MazeEnvironment) -> State:
        confused_flag = 1 if env.confused_turns_remaining > 0 else 0
        dr = planned_step[0] - cell[0]
        dc = planned_step[1] - cell[1]
        return (cell[0], cell[1], dr, dc, confused_flag, fire_time_mod(env))

    def best_action(self, state: State) -> StepDecision:
        q_vals = self._get_q(state)
        return max(STEP_DECISIONS, key=lambda action: q_vals[action.value])

    def select_action(self, state: State, training: bool) -> StepDecision:
        if training and random.random() < self.epsilon:
            return random.choice(STEP_DECISIONS)
        return self.best_action(state)

    def update(
        self,
        state: State,
        action: StepDecision,
        reward: float,
        next_state: Optional[State],
    ) -> None:
        q_sa = self._get_q(state)[action.value]
        q_next = 0.0 if next_state is None else max(self._get_q(next_state))
        td_target = reward + self.gamma * q_next
        self.q_table[state][action.value] += self.alpha * (td_target - q_sa)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    @staticmethod
    def compute_reward(action: StepDecision, outcome: StepOutcome) -> float:
        if outcome.is_goal:
            return REWARD_GOAL
        if outcome.is_dead:
            return REWARD_DEATH

        reward = REWARD_STEP_COST

        if action == StepDecision.WAIT:
            reward += REWARD_WAIT

        if action == StepDecision.GO and outcome.moved:
            reward += REWARD_GO_MOVE

        if outcome.discovered_new_cell:
            reward += REWARD_NEW_CELL

        if outcome.discovered_new_info:
            reward += REWARD_NEW_INFO

        if outcome.wall_hit:
            reward += REWARD_WALL

        if outcome.event == "diverged":
            reward += REWARD_DIVERGE

        return reward

    def save(self, path: str) -> None:
        data = {
            "q_table": {str(state): values for state, values in self.q_table.items()},
            "epsilon": self.epsilon,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        print(f"[Q-learning] Q-table saved -> {path} ({len(self.q_table)} states)")

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.q_table = {
            ast.literal_eval(state): values
            for state, values in data.get("q_table", {}).items()
        }
        self.epsilon = data.get("epsilon", self.epsilon)
        print(f"[Q-learning] Q-table loaded <- {path} ({len(self.q_table)} states)")
        return True


# ============================================================
# BLIND EXPLORER
# ============================================================
class BlindExplorer:
    def __init__(
        self,
        env: MazeEnvironment,
        knowledge: Optional[BlindKnowledge] = None,
        learner: Optional[BlindStepQLearner] = None,
        training: bool = False,
        record_history: bool = True,
    ):
        self.env = env
        self.knowledge = knowledge if knowledge is not None else BlindKnowledge(env.maze_size, env.start, env.goal)
        self.knowledge.start_new_episode()
        self.learner = learner
        self.training = training
        self.record_history = record_history

        self.current: Cell = env.start
        self.physical_steps = 0
        self.deaths = 0
        self.wall_bumps = 0
        self.teleports_taken = 0
        self.confusions_entered = 0
        self.last_new_info_step = 0
        self.history: List[dict] = []
        self._snapshot(event="start", target=None)

    # ---------------- snapshot for animation ----------------
    def _snapshot(self, event: str, target: Optional[Cell]) -> None:
        if not self.record_history:
            return

        self.history.append(
            {
                "current": self.current,
                "visited": set(self.knowledge.visited),
                "blocked_edges": set(self.knowledge.blocked_edges),
                "dangerous": set(self.knowledge.dangerous),
                "confusion": set(self.knowledge.confusion_cells),
                "teleports": dict(self.knowledge.teleport_pairs),
                "active_fire": set(self.env.get_active_fire_cells()),
                "fire_phase": self.env.get_fire_phase(),
                "target": target,
                "event": event,
                "steps": self.physical_steps,
                "deaths": self.deaths,
                "wall_bumps": self.wall_bumps,
                "teleports_taken": self.teleports_taken,
            }
        )

    def _observe_cell(self, cell: Cell) -> bool:
        was_new = cell not in self.knowledge.visited
        self.knowledge.visited.add(cell)
        self.knowledge.tile_of[cell] = int(self.env.obj_matrix[cell])
        return was_new

    def _observe_confusion_if_entered(self, cell: Cell) -> None:
        if int(self.env.obj_matrix[cell]) == CONFUSION:
            self.knowledge.confusion_cells.add(cell)
            self.confusions_entered += 1

    def _time_mod_after_one_action(self) -> int:
        return (self.env.total_actions_executed + 1) % fire_cycle_len(self.env)

    def _mark_new_info(self) -> None:
        self.last_new_info_step = self.physical_steps

    def _log_progress(self, event: str, target: Optional[Cell]) -> None:
        if self.physical_steps % PRINT_EVERY == 0:
            print(
                f"[EXPLORE] steps={self.physical_steps:5d} "
                f"visited={len(self.knowledge.visited):4d} "
                f"walls={len(self.knowledge.blocked_edges) // 2:4d} "
                f"deaths={self.deaths:3d} "
                f"tp={self.teleports_taken:3d} "
                f"target={target} event={event}"
            )

    # ---------------- one physical GO action ----------------
    def _step_physical(self, next_cell: Cell, target: Optional[Cell]) -> StepOutcome:
        if not adjacent(self.current, next_cell):
            return StepOutcome(
                event="diverged",
                moved=False,
                is_dead=False,
                is_goal=False,
                wall_hit=False,
                discovered_new_cell=False,
                discovered_new_info=False,
                teleported=False,
            )

        planned = delta_to_action(self.current, next_cell)
        action_to_send = invert_action(planned) if self.env.confused_turns_remaining > 0 else planned

        before = self.current
        result = self.env.step([action_to_send])
        self.physical_steps += 1

        if result.is_dead:
            death_cell = self.knowledge.teleport_pairs.get(next_cell, next_cell)
            death_phase = self.env.get_fire_phase()
            self.knowledge.dangerous.add(death_cell)
            discovered_new_info = self.knowledge.mark_deadly_phase(death_cell, death_phase)
            if discovered_new_info:
                self._mark_new_info()
            self.current = self.env.start
            self.deaths += 1
            self._snapshot(event="death", target=target)
            return StepOutcome(
                event="dead",
                moved=False,
                is_dead=True,
                is_goal=False,
                wall_hit=False,
                discovered_new_cell=False,
                discovered_new_info=True,
                teleported=result.teleported,
            )

        if result.wall_hits > 0:
            discovered_new_info = self.knowledge.mark_blocked(before, next_cell)
            if discovered_new_info:
                self._mark_new_info()
            self.wall_bumps += 1
            self._snapshot(event="wall", target=target)
            return StepOutcome(
                event="wall",
                moved=False,
                is_dead=False,
                is_goal=False,
                wall_hit=True,
                discovered_new_cell=False,
                discovered_new_info=discovered_new_info,
                teleported=False,
            )

        landing = result.current_position
        discovered_new_info = self.knowledge.mark_open(before, next_cell)
        discovered_new_cell = self._observe_cell(next_cell)
        self._observe_confusion_if_entered(next_cell)

        if result.teleported:
            self.knowledge.teleport_pairs[next_cell] = landing
            self.knowledge.teleport_pairs[landing] = next_cell
            discovered_new_cell = self._observe_cell(landing) or discovered_new_cell
            discovered_new_info = True
            self.teleports_taken += 1

        if discovered_new_cell or discovered_new_info:
            self._mark_new_info()

        self.current = landing
        self._snapshot(event="moved", target=target)
        return StepOutcome(
            event="moved",
            moved=True,
            is_dead=False,
            is_goal=(self.current == self.knowledge.goal or result.is_goal_reached),
            wall_hit=False,
            discovered_new_cell=discovered_new_cell,
            discovered_new_info=discovered_new_info,
            teleported=result.teleported,
        )

    # ---------------- one physical WAIT action ----------------
    def _wait_physical(self, target: Optional[Cell]) -> StepOutcome:
        before = self.current
        result = self.env.step([Action.WAIT])
        self.physical_steps += 1

        if result.is_dead:
            death_phase = self.env.get_fire_phase()
            self.knowledge.dangerous.add(before)
            discovered_new_info = self.knowledge.mark_deadly_phase(before, death_phase)
            if discovered_new_info:
                self._mark_new_info()
            self.current = self.env.start
            self.deaths += 1
            self._snapshot(event="wait-death", target=target)
            return StepOutcome(
                event="dead",
                moved=False,
                is_dead=True,
                is_goal=False,
                wall_hit=False,
                discovered_new_cell=False,
                discovered_new_info=True,
                teleported=result.teleported,
            )

        discovered_new_cell = False
        discovered_new_info = False
        landing = result.current_position

        if result.teleported and landing != before:
            self.knowledge.teleport_pairs[before] = landing
            self.knowledge.teleport_pairs[landing] = before
            discovered_new_cell = self._observe_cell(landing)
            discovered_new_info = True
            self._mark_new_info()
            self.teleports_taken += 1
            self.current = landing
            self._snapshot(event="diverged", target=target)
            return StepOutcome(
                event="diverged",
                moved=True,
                is_dead=False,
                is_goal=(self.current == self.knowledge.goal or result.is_goal_reached),
                wall_hit=False,
                discovered_new_cell=discovered_new_cell,
                discovered_new_info=discovered_new_info,
                teleported=True,
            )

        self.current = before
        self._snapshot(event="wait", target=target)
        return StepOutcome(
            event="wait",
            moved=False,
            is_dead=False,
            is_goal=(self.current == self.knowledge.goal or result.is_goal_reached),
            wall_hit=False,
            discovered_new_cell=False,
            discovered_new_info=False,
            teleported=False,
        )

    def _guarded_step(self, next_cell: Cell, target: Optional[Cell], max_steps: int) -> str:
        wait_chain = 0

        while self.physical_steps < max_steps:
            if self.learner is None:
                return self._step_physical(next_cell, target).event

            state = self.learner.state_for(self.current, next_cell, self.env)
            next_time_mod = self._time_mod_after_one_action()
            go_landing = self.knowledge.teleport_pairs.get(next_cell, next_cell)
            go_deadly = self.knowledge.is_deadly_at_time(go_landing, next_time_mod)
            wait_deadly = self.knowledge.is_deadly_at_time(self.current, next_time_mod)

            if go_deadly and not wait_deadly and wait_chain < MAX_WAIT_CHAIN:
                decision = StepDecision.WAIT
            elif wait_deadly and not go_deadly:
                decision = StepDecision.GO
            elif wait_chain >= MAX_WAIT_CHAIN:
                decision = StepDecision.GO
            else:
                decision = self.learner.select_action(state, training=self.training)

            if decision == StepDecision.WAIT:
                outcome = self._wait_physical(target)
                if outcome.event == "wait":
                    next_state = self.learner.state_for(self.current, next_cell, self.env)
                else:
                    next_state = None
            else:
                outcome = self._step_physical(next_cell, target)
                next_state = None
            if self.training:
                reward = self.learner.compute_reward(decision, outcome)
                self.learner.update(state, decision, reward, next_state)

            if decision == StepDecision.WAIT and outcome.event == "wait":
                wait_chain += 1
                continue

            return outcome.event

        return "budget"

    def _walk_route(
        self,
        route: List[Cell],
        target: Optional[Cell],
        max_steps: int,
        optimistic: bool,
    ) -> str:
        i = 0
        while i < len(route) and self.current == route[i]:
            i += 1

        while i < len(route):
            nxt = route[i]
            if self.current == nxt:
                i += 1
                continue

            step_cell = self.knowledge.find_walk_step(self.current, nxt, optimistic=optimistic)
            if step_cell is None:
                return "blocked"

            event = self._guarded_step(step_cell, target=target, max_steps=max_steps)
            if event == "budget":
                return "budget"
            self._log_progress(event, target)
            if event != "moved":
                return "blocked"
            if self.current == self.knowledge.goal:
                return "goal"

            advanced = False
            while i < len(route) and self.current == route[i]:
                advanced = True
                i += 1
            if not advanced:
                return "blocked"

        return "arrived"

    def _follow_timed_plan(
        self,
        plan: List[PlannedStep],
        target: Optional[Cell],
        max_steps: int,
    ) -> str:
        for planned in plan:
            if self.physical_steps >= max_steps:
                return "budget"

            if planned.step_cell is None:
                outcome = self._wait_physical(target)
            else:
                outcome = self._step_physical(planned.step_cell, target)

            self._log_progress(outcome.event, target)
            if outcome.is_goal or self.current == self.knowledge.goal:
                return "goal"
            if outcome.event not in {"moved", "wait"}:
                return "blocked"

        return "arrived"

    # ---------------- main exploration loop ----------------
    def run(self, max_steps: int = MAX_PHYSICAL_STEPS) -> bool:
        while self.physical_steps < max_steps:
            if self.current == self.knowledge.goal:
                return True

            stalled = (self.physical_steps - self.last_new_info_step) >= STALL_FRONTIER_TRIGGER
            if stalled:
                probe = self.knowledge.pick_frontier_probe(self.current)
                if probe is not None:
                    probe_from, probe_to = probe
                    plan = plan_timed_route(
                        self.knowledge,
                        self.current,
                        fire_time_mod(self.env),
                        probe_from,
                        optimistic=False,
                    )
                    if plan is not None:
                        route_status = self._follow_timed_plan(
                            plan,
                            target=probe_to,
                            max_steps=max_steps,
                        )
                        if route_status == "budget":
                            return False
                        if route_status == "goal":
                            return True

                        if self.current == probe_from:
                            probe_plan = plan_timed_route(
                                self.knowledge,
                                self.current,
                                fire_time_mod(self.env),
                                probe_to,
                                optimistic=True,
                            )
                            if probe_plan is not None:
                                probe_status = self._follow_timed_plan(
                                    probe_plan,
                                    target=probe_to,
                                    max_steps=max_steps,
                                )
                                if probe_status == "budget":
                                    return False
                                if probe_status == "goal":
                                    return True
                                continue

            plan = plan_timed_route(
                self.knowledge,
                self.current,
                fire_time_mod(self.env),
                self.knowledge.goal,
                optimistic=True,
            )
            if plan is None:
                return False
            route_status = self._follow_timed_plan(
                plan,
                target=self.knowledge.goal,
                max_steps=max_steps,
            )
            if route_status == "budget":
                return False
            if route_status == "goal":
                return True

        return self.current == self.knowledge.goal


# ============================================================
# TRAINING / EVALUATION
# ============================================================
def train_blind_step_policy(image_path: str) -> Tuple[BlindStepQLearner, BlindKnowledge, int]:
    qtable_path = qtable_path_for(image_path)
    learner = BlindStepQLearner()
    learner.load(qtable_path)

    env = MazeEnvironment(image_path=image_path, maze_size=MAZE_SIZE)
    learned_knowledge = BlindKnowledge(env.maze_size, env.start, env.goal)
    best_knowledge: Optional[BlindKnowledge] = None
    best_reached = False
    best_score = -1

    print("Training blind GO/WAIT policy...")
    print("  (episodes share learned map/fire memory; the Q-table persists too)\n")

    success_streak = 0
    episodes_ran = 0

    for episode in range(1, TRAIN_EPISODES + 1):
        env.reset()
        explorer = BlindExplorer(
            env,
            knowledge=learned_knowledge,
            learner=learner,
            training=True,
            record_history=False,
        )
        reached = explorer.run(max_steps=TRAINING_STEP_BUDGET)
        learner.decay_epsilon()
        episodes_ran = episode

        if reached:
            success_streak += 1
        else:
            success_streak = 0

        episode_score = len(learned_knowledge.visited) + len(learned_knowledge.blocked_edges) // 2
        if (
            best_knowledge is None
            or (reached and not best_reached)
            or (reached == best_reached and episode_score > best_score)
        ):
            best_knowledge = learned_knowledge.clone()
            best_reached = reached
            best_score = episode_score

        if episode == 1 or episode % TRAIN_PRINT_EVERY == 0 or reached:
            print(
                f"[TRAIN] ep={episode:03d} reached={reached!s:5s} "
                f"steps={explorer.physical_steps:4d} "
                f"mapped={len(learned_knowledge.visited):4d} "
                f"walls={len(learned_knowledge.blocked_edges) // 2:4d} "
                f"deaths={explorer.deaths:3d} "
                f"eps={learner.epsilon:.3f}"
            )

        if success_streak >= SUCCESS_STREAK_TO_STOP:
            print(f"[TRAIN] early stop after {success_streak} consecutive successes")
            break

    learner.save(qtable_path)
    print()
    return learner, best_knowledge if best_knowledge is not None else learned_knowledge.clone(), episodes_ran


def evaluate_policy(
    image_path: str,
    learner: BlindStepQLearner,
    seed_knowledge: Optional[BlindKnowledge] = None,
) -> Tuple[MazeEnvironment, BlindExplorer, bool]:
    env = MazeEnvironment(image_path=image_path, maze_size=MAZE_SIZE)
    env.reset()
    explorer = BlindExplorer(
        env,
        knowledge=seed_knowledge.clone() if seed_knowledge is not None else BlindKnowledge(env.maze_size, env.start, env.goal),
        learner=learner,
        training=False,
        record_history=True,
    )
    reached = explorer.run(max_steps=MAX_PHYSICAL_STEPS)
    return env, explorer, reached


# ============================================================
# ANIMATION
# ============================================================
def build_display(env: MazeEnvironment, snap: dict) -> np.ndarray:
    n = env.maze_size
    disp = np.ones((n, n, 3), dtype=float) * DISPLAY_COLORS[UNKNOWN]

    for r, c in snap["visited"]:
        tile = int(env.obj_matrix[r, c])
        base = DISPLAY_COLORS.get(tile, DISPLAY_COLORS[EMPTY])
        if tile == FIRE:
            base = DISPLAY_COLORS[EMPTY]
        disp[r, c] = base * 0.45 + COL_VISITED * 0.55

    for r, c in snap["dangerous"]:
        disp[r, c] = COL_DANGER

    for r, c in snap["active_fire"]:
        disp[r, c] = DISPLAY_COLORS[FIRE]

    sr, sc = env.start
    gr, gc = env.goal
    disp[sr, sc] = DISPLAY_COLORS[START]
    disp[gr, gc] = DISPLAY_COLORS[GOAL]

    tgt = snap.get("target")
    if tgt is not None:
        disp[tgt[0], tgt[1]] = disp[tgt[0], tgt[1]] * 0.25 + COL_TARGET * 0.75

    cr, cc = snap["current"]
    disp[cr, cc] = COL_AGENT

    return disp


def discovered_walls_to_xy(edges: Set[Tuple[Cell, Cell]]) -> Tuple[List[Optional[int]], List[Optional[int]]]:
    xs: List[Optional[int]] = []
    ys: List[Optional[int]] = []
    drawn = set()

    for a, b in edges:
        key = tuple(sorted([a, b]))
        if key in drawn:
            continue
        drawn.add(key)

        ar, ac = a
        br, bc = b
        if ar == br:
            col = max(ac, bc)
            xs += [col, col, None]
            ys += [ar, ar + 1, None]
        else:
            row = max(ar, br)
            xs += [ac, ac + 1, None]
            ys += [row, row, None]

    return xs, ys


def animate(env: MazeEnvironment, explorer: BlindExplorer, final_path: List[Cell]) -> None:
    history = explorer.history
    n = env.maze_size

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.subplots_adjust(right=0.78)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    im = ax.imshow(
        build_display(env, history[0]),
        extent=(0, n, n, 0),
        interpolation="nearest",
    )

    ax.plot([0, n, n, 0, 0], [0, 0, n, n, 0], color="black", linewidth=1.2)
    wall_lines, = ax.plot([], [], color="black", linewidth=0.9)
    path_line, = ax.plot([], [], color="orange", linewidth=2.2)

    title = ax.set_title("Blind Q-learning exploration", fontsize=10)
    stats_box = ax.text(
        1.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", alpha=0.95),
    )

    state = {"idx": 0, "done": False}
    ani_holder = {"ani": None}

    def draw_final_path_if_any() -> None:
        if not final_path:
            return
        xs_path = [c + 0.5 for _, c in final_path]
        ys_path = [r + 0.5 for r, _ in final_path]
        path_line.set_data(xs_path, ys_path)

    def update(_frame):
        if state["done"]:
            return [im, title, stats_box, wall_lines, path_line]

        state["idx"] = min(len(history) - 1, state["idx"] + SNAPSHOTS_PER_FRAME)
        snap = history[state["idx"]]

        im.set_data(build_display(env, snap))
        xs, ys = discovered_walls_to_xy(snap["blocked_edges"])
        wall_lines.set_data(xs, ys)

        title.set_text(
            f"Blind Q-learning exploration | step {snap['steps']} | "
            f"visited {len(snap['visited'])} | deaths {snap['deaths']} | event {snap['event']}"
        )
        stats_box.set_text(
            "\n".join(
                [
                    f"steps:        {snap['steps']}",
                    f"visited:      {len(snap['visited'])}",
                    f"walls found:  {len(snap['blocked_edges']) // 2}",
                    f"deaths:       {snap['deaths']}",
                    f"wall bumps:   {snap['wall_bumps']}",
                    f"teleports:    {snap['teleports_taken']}",
                    f"fire_phase:   {snap['fire_phase']}",
                    f"event:        {snap['event']}",
                    f"target:       {snap.get('target')}",
                ]
            )
        )

        if state["idx"] >= len(history) - 1:
            state["done"] = True
            draw_final_path_if_any()
            if ani_holder["ani"] is not None:
                ani_holder["ani"].event_source.stop()

        return [im, title, stats_box, wall_lines, path_line]

    total_frames = max(1, len(history) // max(1, SNAPSHOTS_PER_FRAME)) + 20
    ani_holder["ani"] = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=FRAME_MS,
        blit=False,
        repeat=False,
    )
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    preview_env = MazeEnvironment(image_path=IMAGE_PATH, maze_size=MAZE_SIZE)
    print(f"Start: {preview_env.start}")
    print(f"Goal : {preview_env.goal}\n")

    learner, training_knowledge, episodes_ran = train_blind_step_policy(IMAGE_PATH)

    print("Running learned-memory evaluation...")
    print("  (the learned map/fire memory is reused here, so this measures actual retained learning)\n")

    env, explorer, reached = evaluate_policy(IMAGE_PATH, learner, seed_knowledge=training_knowledge)

    if reached:
        print("GOAL REACHED")
    else:
        print("GOAL NOT REACHED (evaluation stayed blind but ran out of progress or budget)")

    print(f"Training episodes     : {episodes_ran}")
    print(f"Q-table states        : {len(learner.q_table)}")
    print(f"Best train map size   : {len(training_knowledge.visited)}")
    print(f"Physical steps        : {explorer.physical_steps}")
    print(f"Cells discovered      : {len(explorer.knowledge.visited)}")
    print(f"Walls discovered      : {len(explorer.knowledge.blocked_edges) // 2}")
    print(f"Deaths in fire        : {explorer.deaths}")
    print(f"Wall bumps            : {explorer.wall_bumps}")
    print(f"Teleports taken       : {explorer.teleports_taken}")
    print(f"Confusion tiles hit   : {explorer.confusions_entered}")

    final_path = shortest_path_in_discovered(explorer.knowledge) if reached else []
    if final_path:
        print(f"Shortest path in discovered graph : {len(final_path) - 1} steps")

    animate(env, explorer, final_path)


if __name__ == "__main__":
    main()
