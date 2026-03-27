from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

Cell = Tuple[int, int]

# ============================================================
# TILE TYPES
# ============================================================
EMPTY = 0
FIRE = 1
CONFUSION = 2
TP_PURPLE = 3
TP_RED = 4
TP_GREEN = 5
START = 6
GOAL = 7
UNKNOWN = 99

TARGET_COLORS = {
    FIRE:       (255, 145, 76),
    CONFUSION:  (255, 222, 89),
    TP_PURPLE:  (140, 82, 255),
    TP_RED:     (255, 49, 50),
    TP_GREEN:   (1, 191, 99),
    START:      (15, 192, 223),
    GOAL:       (0, 74, 173),
}

COLOR_TOL = 45


# ============================================================
# ACTIONS / RESULT
# ============================================================
class Action(Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    WAIT = 4


@dataclass
class TurnResult:
    wall_hits: int = 0
    current_position: Cell = (0, 0)
    is_dead: bool = False
    is_confused: bool = False
    is_goal_reached: bool = False
    teleported: bool = False
    actions_executed: int = 0


@dataclass
class Icon:
    x: int
    y: int
    w: int
    h: int
    cx: float
    cy: float
    row: int
    col: int
    kind: int
    rgb_mean: Tuple[float, float, float]


# ============================================================
# IMAGE / MAZE PARSING
# ============================================================
def load_image_rgb(path: str) -> np.ndarray:
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def infer_grid_step(gray: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
    black = gray < 40
    colsum = black.sum(axis=0)
    rowsum = black.sum(axis=1)

    col_peaks = np.where(colsum > gray.shape[0] * 0.25)[0]
    row_peaks = np.where(rowsum > gray.shape[1] * 0.25)[0]

    def compress_runs(arr: np.ndarray) -> np.ndarray:
        if len(arr) == 0:
            return np.array([])
        groups = [[arr[0]]]
        for v in arr[1:]:
            if v == groups[-1][-1] + 1:
                groups[-1].append(v)
            else:
                groups.append([v])
        return np.array([int(np.mean(g)) for g in groups])

    col_lines = compress_runs(col_peaks)
    row_lines = compress_runs(row_peaks)

    if len(col_lines) < 2 or len(row_lines) < 2:
        raise RuntimeError("Could not infer maze grid lines.")

    step_x = int(round(np.median(np.diff(col_lines))))
    step_y = int(round(np.median(np.diff(row_lines))))
    step = int(round((step_x + step_y) / 2))
    return step, col_lines, row_lines


def build_wall_matrices(gray: np.ndarray, step: int, n: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    h, w = gray.shape
    wall_mask = gray < 60

    line_pos = np.arange(0, n + 1) * step
    line_pos = np.clip(line_pos, 0, min(h, w) - 1)

    vertical_walls = np.zeros((n, n + 1), dtype=np.uint8)
    horizontal_walls = np.zeros((n + 1, n), dtype=np.uint8)

    for r in range(n):
        y0 = int(r * step + step * 0.2)
        y1 = int((r + 1) * step - step * 0.2)
        y0 = max(0, y0)
        y1 = min(h - 1, y1)
        for c in range(n + 1):
            x = int(line_pos[c])
            x0 = max(0, x - 1)
            x1 = min(w - 1, x + 1)
            patch = wall_mask[y0:y1 + 1, x0:x1 + 1]
            vertical_walls[r, c] = 1 if patch.mean() > 0.35 else 0

    for r in range(n + 1):
        y = int(line_pos[r])
        y0 = max(0, y - 1)
        y1 = min(h - 1, y + 1)
        for c in range(n):
            x0 = int(c * step + step * 0.2)
            x1 = int((c + 1) * step - step * 0.2)
            x0 = max(0, x0)
            x1 = min(w - 1, x1)
            patch = wall_mask[y0:y1 + 1, x0:x1 + 1]
            horizontal_walls[r, c] = 1 if patch.mean() > 0.35 else 0

    return vertical_walls, horizontal_walls


def color_distance(c1, c2) -> float:
    c1 = np.array(c1, dtype=np.float32)
    c2 = np.array(c2, dtype=np.float32)
    return float(np.linalg.norm(c1 - c2))


def classify_icon(rgb_mean: Tuple[float, float, float]) -> int:
    best_kind = UNKNOWN
    best_dist = float("inf")

    for kind, target_rgb in TARGET_COLORS.items():
        dist = color_distance(rgb_mean, target_rgb)
        if dist < best_dist:
            best_dist = dist
            best_kind = kind

    return best_kind if best_dist <= COLOR_TOL else UNKNOWN


def detect_colored_icons(img_rgb: np.ndarray, step: int, maze_size: int = 64) -> List[Icon]:
    maxc = img_rgb.max(axis=2)
    minc = img_rgb.min(axis=2)
    sat = maxc - minc

    color_mask = ((sat > 40) & (maxc > 60)).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(color_mask, connectivity=8)

    icons: List[Icon] = []
    for i in range(1, n_labels):
        x, y, bw, bh, area = stats[i]
        if area < 8:
            continue
        if bw > step * 2 or bh > step * 2:
            continue

        patch_mask = labels[y:y+bh, x:x+bw] == i
        patch_rgb = img_rgb[y:y+bh, x:x+bw][patch_mask]
        rgb_mean = tuple(np.mean(patch_rgb, axis=0))

        cx, cy = centroids[i]
        col = int(np.clip(round((cx - step / 2) / step), 0, maze_size - 1))
        row = int(np.clip(round((cy - step / 2) / step), 0, maze_size - 1))
        kind = classify_icon(rgb_mean)

        icons.append(
            Icon(
                x=int(x), y=int(y), w=int(bw), h=int(bh),
                cx=float(cx), cy=float(cy),
                row=row, col=col, kind=kind,
                rgb_mean=tuple(float(v) for v in rgb_mean),
            )
        )
    return icons


def build_object_matrix(icons: List[Icon], n: int = 64) -> np.ndarray:
    obj = np.zeros((n, n), dtype=np.int32)
    for icon in icons:
        obj[icon.row, icon.col] = icon.kind
    return obj


def find_single_cell(obj_matrix: np.ndarray, target_value: int, name: str) -> Cell:
    cells = list(zip(*np.where(obj_matrix == target_value)))
    if len(cells) != 1:
        raise ValueError(f"Expected exactly 1 {name}, found {len(cells)}")
    return cells[0]


def build_teleport_pairs(obj_matrix: np.ndarray) -> Dict[Cell, Cell]:
    teleport_pairs: Dict[Cell, Cell] = {}
    for tp_kind in [TP_PURPLE, TP_RED, TP_GREEN]:
        cells = list(zip(*np.where(obj_matrix == tp_kind)))
        cells = sorted(cells)
        if len(cells) < 2:
            continue
        if len(cells) == 2:
            a, b = cells
            teleport_pairs[a] = b
            teleport_pairs[b] = a
        else:
            for i in range(len(cells)):
                teleport_pairs[cells[i]] = cells[(i + 1) % len(cells)]
    return teleport_pairs


def extract_fire_cells_from_image(path: str, step: int, maze_size: int = 64) -> Set[Cell]:
    img_rgb = load_image_rgb(path)
    icons = detect_colored_icons(img_rgb, step, maze_size=maze_size)
    obj = build_object_matrix(icons, n=maze_size)
    return set(zip(*np.where(obj == FIRE)))


# ============================================================
# MAZE ENVIRONMENT
# ============================================================
class MazeEnvironment:
    def __init__(
        self,
        image_path: str,
        fire_phase_images: Optional[List[str]] = None,
        maze_size: int = 64,
    ):
        self.image_path = image_path
        self.fire_phase_images = fire_phase_images or [image_path]
        self.maze_size = maze_size

        img_rgb = load_image_rgb(image_path)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        self.step_px, _, _ = infer_grid_step(gray)
        self.vertical_walls, self.horizontal_walls = build_wall_matrices(gray, self.step_px, n=maze_size)

        icons = detect_colored_icons(img_rgb, self.step_px, maze_size=maze_size)
        self.obj_matrix = build_object_matrix(icons, n=maze_size)

        self.start = find_single_cell(self.obj_matrix, START, "start")
        self.goal = find_single_cell(self.obj_matrix, GOAL, "goal")
        self.teleport_pairs = build_teleport_pairs(self.obj_matrix)

        self.fire_phase_sets = [
            extract_fire_cells_from_image(path, self.step_px, maze_size=maze_size)
            for path in self.fire_phase_images
        ]
        if not self.fire_phase_sets:
            self.fire_phase_sets = [set()]

        self.position: Cell = self.start
        self.turns_taken = 0
        self.deaths = 0
        self.confused_count = 0
        self.cells_visited: List[Cell] = []
        self.unique_cells: Set[Cell] = set()
        self.goal_reached = False

        self.confused_this_turn = False
        self.confused_turns_remaining = 0
        self.total_actions_executed = 0

    def reset(self) -> Cell:
        self.position = self.start
        self.turns_taken = 0
        self.deaths = 0
        self.confused_count = 0
        self.cells_visited = [self.start]
        self.unique_cells = {self.start}
        self.goal_reached = False
        self.confused_this_turn = False
        self.confused_turns_remaining = 0
        self.total_actions_executed = 0
        return self.start

    def get_active_fire_cells(self) -> Set[Cell]:
        phase = (self.total_actions_executed // 5) % len(self.fire_phase_sets)
        return self.fire_phase_sets[phase]

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.maze_size and 0 <= c < self.maze_size

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

    def apply_confusion(self, action: Action) -> Action:
        if action == Action.MOVE_UP:
            return Action.MOVE_DOWN
        if action == Action.MOVE_DOWN:
            return Action.MOVE_UP
        if action == Action.MOVE_LEFT:
            return Action.MOVE_RIGHT
        if action == Action.MOVE_RIGHT:
            return Action.MOVE_LEFT
        return Action.WAIT

    def action_to_target(self, pos: Cell, action: Action) -> Cell:
        r, c = pos
        if action == Action.MOVE_UP:
            return (r - 1, c)
        if action == Action.MOVE_DOWN:
            return (r + 1, c)
        if action == Action.MOVE_LEFT:
            return (r, c - 1)
        if action == Action.MOVE_RIGHT:
            return (r, c + 1)
        return pos

    def _apply_tile_effects(self, result: TurnResult) -> bool:
        if self.position in self.get_active_fire_cells():
            result.is_dead = True
            result.current_position = self.position
            self.deaths += 1
            return True

        tile = self.obj_matrix[self.position[0], self.position[1]]

        if tile == CONFUSION:
            self.confused_count += 1
            self.confused_turns_remaining = max(self.confused_turns_remaining, 2)
            self.confused_this_turn = True

        if self.position in self.teleport_pairs:
            self.position = self.teleport_pairs[self.position]
            result.teleported = True
            result.current_position = self.position

            if self.position in self.get_active_fire_cells():
                result.is_dead = True
                self.deaths += 1
                return True

        if self.position == self.goal or tile == GOAL:
            result.is_goal_reached = True
            self.goal_reached = True
            result.current_position = self.position
            return True

        return False

    def step_one_action(self, action: Action, turn_confused: bool) -> TurnResult:
        result = TurnResult(current_position=self.position)

        effective_action = self.apply_confusion(action) if turn_confused else action
        target = self.action_to_target(self.position, effective_action)

        if effective_action != Action.WAIT and not self.can_move(self.position, target):
            result.wall_hits = 1
            result.actions_executed = 1
            self.total_actions_executed += 1
            result.current_position = self.position
            result.is_confused = turn_confused or self.confused_this_turn
            return result

        self.position = target
        result.actions_executed = 1
        self.total_actions_executed += 1
        result.current_position = self.position
        self.cells_visited.append(self.position)
        self.unique_cells.add(self.position)

        self._apply_tile_effects(result)
        result.is_confused = turn_confused or self.confused_this_turn
        return result

    def finish_turn(self, result: TurnResult) -> TurnResult:
        self.turns_taken += 1

        if result.is_dead:
            self.position = self.start
            result.current_position = self.start

        if self.confused_turns_remaining > 0:
            self.confused_turns_remaining -= 1

        return result

    def step(self, actions: List[Action]) -> TurnResult:
        if not (1 <= len(actions) <= 5):
            raise ValueError("actions must contain between 1 and 5 actions")

        final_result = TurnResult(current_position=self.position)

        turn_confused = self.confused_turns_remaining > 0
        self.confused_this_turn = turn_confused

        for action in actions:
            one = self.step_one_action(action, turn_confused)

            final_result.wall_hits += one.wall_hits
            final_result.current_position = one.current_position
            final_result.is_dead = one.is_dead
            final_result.is_confused = one.is_confused
            final_result.is_goal_reached = one.is_goal_reached
            final_result.teleported = final_result.teleported or one.teleported
            final_result.actions_executed += one.actions_executed

            if one.is_dead or one.is_goal_reached:
                break

        return self.finish_turn(final_result)

    def get_episode_stats(self) -> dict:
        return {
            "turns_taken": self.turns_taken,
            "deaths": self.deaths,
            "confused": self.confused_count,
            "cells_explored": len(self.unique_cells),
            "goal_reached": self.goal_reached,
            "total_actions_executed": self.total_actions_executed,
        }