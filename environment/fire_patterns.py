from __future__ import annotations

from typing import List, Set

from .image_parsing import detect_cells_by_color_targets, load_image_rgb
from .models import Cell, FIRE, FIRE_CENTER, TARGET_COLORS


def extract_fire_cells_from_image(path: str, step: int, maze_size: int = 64) -> Set[Cell]:
    img_rgb = load_image_rgb(path)
    fire_cells = detect_cells_by_color_targets(img_rgb, step, TARGET_COLORS[FIRE], maze_size=maze_size)
    fire_cells |= detect_cells_by_color_targets(img_rgb, step, TARGET_COLORS[FIRE_CENTER], maze_size=maze_size)
    return fire_cells


def split_fire_components(cells: Set[Cell]) -> List[Set[Cell]]:
    cells = set(cells)
    components = []
    seen = set()

    for start in cells:
        if start in seen:
            continue

        stack = [start]
        component = set()
        seen.add(start)

        while stack:
            row, col = stack.pop()
            component.add((row, col))

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    neighbor = (row + dr, col + dc)
                    if neighbor in cells and neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)

        components.append(component)

    return components


def find_fire_root(component: Set[Cell], explicit_roots: Set[Cell] | None = None) -> Cell:
    component = set(component)
    if explicit_roots:
        roots_in_component = sorted(component & explicit_roots)
        if roots_in_component:
            return roots_in_component[0]

    candidates = []
    for row, col in component:
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                neighbor = (row + dr, col + dc)
                if neighbor in component:
                    neighbors.append((dr, dc))

        if len(neighbors) == 2:
            first, second = neighbors
            if not (first[0] == -second[0] and first[1] == -second[1]):
                candidates.append((row, col))

    if len(candidates) == 1:
        return candidates[0]

    center_row = sum(row for row, _ in component) / len(component)
    center_col = sum(col for _, col in component) / len(component)
    return min(component, key=lambda cell: (cell[0] - center_row) ** 2 + (cell[1] - center_col) ** 2)


def rotate_point_about_root_90_clockwise(point: Cell, root: Cell) -> Cell:
    point_row, point_col = point
    root_row, root_col = root
    delta_row = point_row - root_row
    delta_col = point_col - root_col
    return (root_row + delta_col, root_col - delta_row)


def rotate_component_about_root(component: Set[Cell], root: Cell, quarter_turns: int, n: int) -> Set[Cell]:
    rotated = set(component)
    turns = quarter_turns % 4

    for _ in range(turns):
        rotated = {rotate_point_about_root_90_clockwise(point, root) for point in rotated}

    return {(row, col) for row, col in rotated if 0 <= row < n and 0 <= col < n}


def build_rotating_fire_phase_sets(
    base_fire_cells: Set[Cell],
    n: int,
    fire_center_cells: Set[Cell] | None = None,
) -> List[Set[Cell]]:
    components = split_fire_components(base_fire_cells)
    roots = [find_fire_root(component, fire_center_cells) for component in components]

    phases = []
    for quarter_turn in range(4):
        phase_cells = set()
        for component, root in zip(components, roots):
            phase_cells |= rotate_component_about_root(component, root, quarter_turn, n)
        phases.append(phase_cells)

    return phases
