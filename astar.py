from __future__ import annotations

import heapq
from typing import Callable, Dict, List, Tuple, Set, Any

Cell = Tuple[int, int]


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct_path(came_from: Dict[Cell, Cell], current: Cell) -> List[Cell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar_search(
    start: Cell,
    goal: Cell,
    neighbors_fn: Callable[[Cell], List[Cell]],
    heuristic_fn: Callable[[Cell, Cell], int] = manhattan,
) -> List[Cell]:
    result = astar_search_debug(start, goal, neighbors_fn, heuristic_fn)
    return result["path"]


def astar_search_debug(
    start: Cell,
    goal: Cell,
    neighbors_fn: Callable[[Cell], List[Cell]],
    heuristic_fn: Callable[[Cell, Cell], int] = manhattan,
) -> Dict[str, Any]:
    if start == goal:
        return {
            "path": [start],
            "expanded_order": [start],
            "closed_set": {start},
            "g_score": {start: 0},
        }

    open_heap: List[Tuple[int, int, Cell]] = []
    heapq.heappush(open_heap, (heuristic_fn(start, goal), 0, start))

    came_from: Dict[Cell, Cell] = {}
    g_score: Dict[Cell, int] = {start: 0}
    closed: Set[Cell] = set()
    expanded_order: List[Cell] = []

    while open_heap:
        _, g, current = heapq.heappop(open_heap)

        if current in closed:
            continue

        closed.add(current)
        expanded_order.append(current)

        if current == goal:
            return {
                "path": reconstruct_path(came_from, current),
                "expanded_order": expanded_order,
                "closed_set": closed,
                "g_score": g_score,
            }

        for nb in neighbors_fn(current):
            tentative_g = g + 1
            if tentative_g < g_score.get(nb, float("inf")):
                g_score[nb] = tentative_g
                came_from[nb] = current
                f = tentative_g + heuristic_fn(nb, goal)
                heapq.heappush(open_heap, (f, tentative_g, nb))

    return {
        "path": [],
        "expanded_order": expanded_order,
        "closed_set": closed,
        "g_score": g_score,
    }