from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .config import FIRE_PHASE_COUNT, FIRE_PHASE_TICKS
from .knowledge import BlindKnowledge, Cell


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def distances_from(knowledge: BlindKnowledge, start: Cell) -> Dict[Cell, int]:
    distances = {start: 0}
    queue = deque([start])

    while queue:
        current = queue.popleft()
        for neighbor in knowledge.planning_neighbors(current):
            if neighbor in distances:
                continue
            distances[neighbor] = distances[current] + 1
            queue.append(neighbor)

    return distances


def pick_frontier_probe(knowledge: BlindKnowledge, current: Cell) -> Optional[Tuple[Cell, Cell]]:
    distances = distances_from(knowledge, current)
    best_probe = None
    best_key = None

    for visited_cell in knowledge.visited:
        if visited_cell not in distances:
            continue

        for neighbor in knowledge.candidate_neighbors(visited_cell):
            if not knowledge.is_probe_candidate(visited_cell, neighbor):
                continue

            goal_h = manhattan(neighbor, knowledge.goal)
            f_score = distances[visited_cell] + goal_h
            key = (f_score, goal_h, distances[visited_cell])
            if best_key is None or key < best_key:
                best_key = key
                best_probe = (visited_cell, neighbor)

    return best_probe


def shortest_path_in_discovered(knowledge: BlindKnowledge) -> List[Cell]:
    queue = deque([knowledge.start])
    came_from: Dict[Cell, Cell] = {knowledge.start: knowledge.start}

    while queue:
        current = queue.popleft()
        if current == knowledge.goal:
            path = [current]
            while came_from[current] != current:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for neighbor in knowledge.planning_neighbors(current):
            if neighbor in came_from:
                continue
            came_from[neighbor] = current
            queue.append(neighbor)

    return []


@dataclass(frozen=True)
class PlannedStep:
    step_cell: Optional[Cell]
    landing: Cell


def plan_timed_route(
    knowledge: BlindKnowledge,
    start: Cell,
    start_time_mod: int,
    target: Cell,
    optimistic: bool,
) -> Optional[List[PlannedStep]]:
    if start == target:
        return []

    cycle_len = FIRE_PHASE_TICKS * FIRE_PHASE_COUNT
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
            current = state
            while current in came_from:
                previous, step = came_from[current]
                plan.append(step)
                current = previous
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
