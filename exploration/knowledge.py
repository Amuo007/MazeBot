from __future__ import annotations

from typing import Dict, List, Set, Tuple

from environment import Action, MazeEnvironment, START

from .config import FIRE_PHASE_TICKS

Cell = Tuple[int, int]


def adjacent(a: Cell, b: Cell) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


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


def fire_cycle_len(env: MazeEnvironment) -> int:
    return max(1, len(env.fire_phase_sets) * FIRE_PHASE_TICKS)


def fire_time_mod(env: MazeEnvironment) -> int:
    return env.total_actions_executed % fire_cycle_len(env)


def fire_phase_for_time_mod(time_mod: int) -> int:
    return time_mod // FIRE_PHASE_TICKS


class BlindKnowledge:
    def __init__(self, maze_size: int, start: Cell, goal: Cell):
        self.n = maze_size
        self.start = start
        self.goal = goal

        self.visited: Set[Cell] = {start}
        self.tile_of: Dict[Cell, int] = {start: START}

        self.open_edges: Set[Tuple[Cell, Cell]] = set()
        self.blocked_edges: Set[Tuple[Cell, Cell]] = set()
        self.teleport_pairs: Dict[Cell, Cell] = {}

        self.dangerous: Set[Cell] = set()
        self.deadly_phases: Dict[Cell, Set[int]] = {}
        self.abandoned: Set[Cell] = set()
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

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.n and 0 <= c < self.n

    def candidate_neighbors(self, cell: Cell) -> List[Cell]:
        r, c = cell
        return [neighbor for neighbor in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)] if self.in_bounds(neighbor)]

    def open_neighbors(self, cell: Cell) -> List[Cell]:
        return [neighbor for neighbor in self.candidate_neighbors(cell) if (cell, neighbor) in self.open_edges]

    def planning_neighbors(self, cell: Cell) -> List[Cell]:
        return [self.teleport_pairs.get(neighbor, neighbor) for neighbor in self.open_neighbors(cell)]

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

    def is_probe_candidate(self, source: Cell, neighbor: Cell) -> bool:
        if neighbor in self.visited:
            return False
        if (source, neighbor) in self.blocked_edges:
            return False
        if neighbor in self.abandoned:
            return False
        return True
