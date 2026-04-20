from __future__ import annotations

from typing import Dict, List, Set, Tuple

import cv2
import numpy as np

from .models import COLOR_TOL, TELEPORT_TILES, TARGET_COLORS, Cell, Icon, UNKNOWN


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
        for value in arr[1:]:
            if value == groups[-1][-1] + 1:
                groups[-1].append(value)
            else:
                groups.append([value])
        return np.array([int(np.mean(group)) for group in groups])

    col_lines = compress_runs(col_peaks)
    row_lines = compress_runs(row_peaks)

    if len(col_lines) < 2 or len(row_lines) < 2:
        raise RuntimeError("Could not infer maze grid lines.")

    step_x = int(round(np.median(np.diff(col_lines))))
    step_y = int(round(np.median(np.diff(row_lines))))
    step = int(round((step_x + step_y) / 2))
    return step, col_lines, row_lines


def build_wall_matrices(gray: np.ndarray, step: int, n: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    height, width = gray.shape
    wall_mask = gray < 60

    line_pos = np.arange(0, n + 1) * step
    line_pos = np.clip(line_pos, 0, min(height, width) - 1)

    vertical_walls = np.zeros((n, n + 1), dtype=np.uint8)
    horizontal_walls = np.zeros((n + 1, n), dtype=np.uint8)

    for row in range(n):
        y0 = int(row * step + step * 0.2)
        y1 = int((row + 1) * step - step * 0.2)
        y0 = max(0, y0)
        y1 = min(height - 1, y1)
        for col in range(n + 1):
            x = int(line_pos[col])
            x0 = max(0, x - 1)
            x1 = min(width - 1, x + 1)
            patch = wall_mask[y0:y1 + 1, x0:x1 + 1]
            vertical_walls[row, col] = 1 if patch.mean() > 0.35 else 0

    for row in range(n + 1):
        y = int(line_pos[row])
        y0 = max(0, y - 1)
        y1 = min(height - 1, y + 1)
        for col in range(n):
            x0 = int(col * step + step * 0.2)
            x1 = int((col + 1) * step - step * 0.2)
            x0 = max(0, x0)
            x1 = min(width - 1, x1)
            patch = wall_mask[y0:y1 + 1, x0:x1 + 1]
            horizontal_walls[row, col] = 1 if patch.mean() > 0.35 else 0

    return vertical_walls, horizontal_walls


def color_distance(c1, c2) -> float:
    color_1 = np.array(c1, dtype=np.float32)
    color_2 = np.array(c2, dtype=np.float32)
    return float(np.linalg.norm(color_1 - color_2))


def min_color_distance(rgb_mean: Tuple[float, float, float], target_rgbs: List[Tuple[int, int, int]]) -> float:
    return min(color_distance(rgb_mean, target_rgb) for target_rgb in target_rgbs)


def classify_icon(rgb_mean: Tuple[float, float, float]) -> int:
    best_kind = UNKNOWN
    best_distance = float("inf")

    for kind, target_rgbs in TARGET_COLORS.items():
        distance = min_color_distance(rgb_mean, target_rgbs)
        if distance < best_distance:
            best_distance = distance
            best_kind = kind

    return best_kind if best_distance <= COLOR_TOL else UNKNOWN


def detect_cells_by_color_targets(
    img_rgb: np.ndarray,
    step: int,
    target_rgbs: List[Tuple[int, int, int]],
    maze_size: int = 64,
    sample_margin: float = 0.22,
    min_color_pixels: int = 6,
) -> Set[Cell]:
    cells: Set[Cell] = set()
    height, width, _ = img_rgb.shape

    for row in range(maze_size):
        y0 = int(row * step + step * sample_margin)
        y1 = int((row + 1) * step - step * sample_margin)
        y0 = max(0, min(height - 1, y0))
        y1 = max(0, min(height - 1, y1))
        if y1 < y0:
            continue

        for col in range(maze_size):
            x0 = int(col * step + step * sample_margin)
            x1 = int((col + 1) * step - step * sample_margin)
            x0 = max(0, min(width - 1, x0))
            x1 = max(0, min(width - 1, x1))
            if x1 < x0:
                continue

            patch = img_rgb[y0:y1 + 1, x0:x1 + 1]
            if patch.size == 0:
                continue

            maxc = patch.max(axis=2)
            minc = patch.min(axis=2)
            sat = maxc - minc
            color_mask = (sat > 40) & (maxc > 60)
            if int(color_mask.sum()) < min_color_pixels:
                continue

            rgb_mean = tuple(np.mean(patch[color_mask], axis=0))
            if min_color_distance(rgb_mean, target_rgbs) <= COLOR_TOL:
                cells.add((row, col))

    return cells


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
    for index in range(1, n_labels):
        x, y, box_width, box_height, area = stats[index]
        if area < 8:
            continue
        if box_width > step * 2 or box_height > step * 2:
            continue

        patch_mask = labels[y:y + box_height, x:x + box_width] == index
        patch_rgb = img_rgb[y:y + box_height, x:x + box_width][patch_mask]
        rgb_mean = tuple(np.mean(patch_rgb, axis=0))

        cx, cy = centroids[index]
        col = int(np.clip(round((cx - step / 2) / step), 0, maze_size - 1))
        row = int(np.clip(round((cy - step / 2) / step), 0, maze_size - 1))
        kind = classify_icon(rgb_mean)

        icons.append(
            Icon(
                x=int(x),
                y=int(y),
                w=int(box_width),
                h=int(box_height),
                cx=float(cx),
                cy=float(cy),
                row=row,
                col=col,
                kind=kind,
                rgb_mean=tuple(float(value) for value in rgb_mean),
            )
        )

    return icons


def build_object_matrix(icons: List[Icon], n: int = 64) -> np.ndarray:
    obj_matrix = np.zeros((n, n), dtype=np.int32)
    for icon in icons:
        obj_matrix[icon.row, icon.col] = icon.kind
    return obj_matrix


def overlay_precise_cells(
    obj_matrix: np.ndarray,
    img_rgb: np.ndarray,
    step: int,
    kinds: List[int],
    maze_size: int = 64,
) -> None:
    for kind in kinds:
        cells = detect_cells_by_color_targets(img_rgb, step, TARGET_COLORS[kind], maze_size=maze_size)
        for cell in cells:
            obj_matrix[cell] = kind


def find_single_cell(obj_matrix: np.ndarray, target_value: int, name: str) -> Cell:
    cells = list(zip(*np.where(obj_matrix == target_value)))
    if len(cells) != 1:
        raise ValueError(f"Expected exactly 1 {name}, found {len(cells)}")
    return cells[0]


def build_teleport_pairs(obj_matrix: np.ndarray) -> Dict[Cell, Cell]:
    teleport_pairs: Dict[Cell, Cell] = {}
    for teleport_kind in TELEPORT_TILES:
        cells = sorted(list(zip(*np.where(obj_matrix == teleport_kind))))
        if len(cells) < 2:
            continue
        if len(cells) == 2:
            a, b = cells
            teleport_pairs[a] = b
            teleport_pairs[b] = a
        else:
            for index in range(len(cells)):
                teleport_pairs[cells[index]] = cells[(index + 1) % len(cells)]
    return teleport_pairs
