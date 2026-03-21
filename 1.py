import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dataclasses import dataclass

# ============================================================
# CONFIG
# ============================================================
IMAGE_PATH = "maze_5.png"   # <- change to your PNG filename
SHOW_DEBUG = True
LIVE_REFRESH_MS = 200
MAZE_SIZE = 64

# color tolerance: increase if your painted colors vary a bit
COLOR_TOL = 45

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

NAME_TO_CHAR = {
    EMPTY: ".",
    FIRE: "F",
    CONFUSION: "C",
    TP_PURPLE: "P",
    TP_RED: "R",
    TP_GREEN: "G",
    START: "S",
    GOAL: "E",
    UNKNOWN: "?"
}

# exact RGB colors from your painted version
TARGET_COLORS = {
    FIRE:       (255, 145, 76),   # #ff914c
    CONFUSION:  (255, 222, 89),   # #ffde59
    TP_PURPLE:  (140, 82, 255),   # #8c52ff
    TP_RED:     (255, 49, 50),    # #ff3132
    TP_GREEN:   (1, 191, 99),     # #01bf63
    START:      (15, 192, 223),   # #0fc0df
    GOAL:       (0, 74, 173),     # #004aad
}

# ============================================================
# DATA STRUCTURES
# ============================================================
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
    rgb_mean: tuple

# ============================================================
# IMAGE HELPERS
# ============================================================
def load_image(path):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def infer_grid_step(gray):
    black = gray < 40
    colsum = black.sum(axis=0)
    rowsum = black.sum(axis=1)

    col_peaks = np.where(colsum > gray.shape[0] * 0.25)[0]
    row_peaks = np.where(rowsum > gray.shape[1] * 0.25)[0]

    def compress_runs(arr):
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

def build_wall_matrices(gray, step, n=64):
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

# ============================================================
# COLOR MATCHING
# ============================================================
def color_distance(c1, c2):
    c1 = np.array(c1, dtype=np.float32)
    c2 = np.array(c2, dtype=np.float32)
    return np.linalg.norm(c1 - c2)

def classify_icon(rgb_mean):
    best_kind = UNKNOWN
    best_dist = float("inf")

    for kind, target_rgb in TARGET_COLORS.items():
        dist = color_distance(rgb_mean, target_rgb)
        if dist < best_dist:
            best_dist = dist
            best_kind = kind

    if best_dist <= COLOR_TOL:
        return best_kind
    return UNKNOWN

# ============================================================
# ICON DETECTION
# ============================================================
def detect_colored_icons(img_rgb, step):
    """
    Detect colored painted blobs and classify them by nearest known RGB.
    """
    maxc = img_rgb.max(axis=2)
    minc = img_rgb.min(axis=2)
    sat = maxc - minc

    # detect colorful things, ignore white background and black walls
    color_mask = ((sat > 40) & (maxc > 60)).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        color_mask, connectivity=8
    )

    icons = []
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

        col = int(np.clip(round((cx - step / 2) / step), 0, MAZE_SIZE - 1))
        row = int(np.clip(round((cy - step / 2) / step), 0, MAZE_SIZE - 1))

        kind = classify_icon(rgb_mean)

        icons.append(
            Icon(
                x=int(x),
                y=int(y),
                w=int(bw),
                h=int(bh),
                cx=float(cx),
                cy=float(cy),
                row=int(row),
                col=int(col),
                kind=int(kind),
                rgb_mean=tuple(float(v) for v in rgb_mean),
            )
        )

    return icons, color_mask

# ============================================================
# MAZE MATRIX BUILD
# ============================================================
def build_object_matrix(icons, n=64):
    obj = np.zeros((n, n), dtype=np.int32)
    for icon in icons:
        obj[icon.row, icon.col] = icon.kind
    return obj

def print_symbol_matrix(obj):
    lines = []
    for r in range(obj.shape[0]):
        line = "".join(NAME_TO_CHAR.get(v, "?") for v in obj[r])
        lines.append(line)
    return "\n".join(lines)

# ============================================================
# LIVE MATPLOTLIB VIEW
# ============================================================
def draw_live_map(vertical_walls, horizontal_walls, obj_matrix, icons, step):
    n = obj_matrix.shape[0]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("white")
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    ax.set_title("Maze matrix live view")

    for r in range(n):
        for c in range(n + 1):
            if vertical_walls[r, c]:
                ax.plot([c, c], [r, r + 1], color="black", linewidth=1)

    for r in range(n + 1):
        for c in range(n):
            if horizontal_walls[r, c]:
                ax.plot([c, c + 1], [r, r], color="black", linewidth=1)

    for icon in icons:
        rr, cc = icon.row, icon.col
        label = NAME_TO_CHAR.get(icon.kind, "?")

        ax.add_patch(
            Rectangle(
                (cc + 0.15, rr + 0.15),
                0.7,
                0.7,
                fill=False,
                edgecolor="red",
                linewidth=1,
            )
        )
        ax.text(cc + 0.5, rr + 0.55, label, ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.pause(LIVE_REFRESH_MS / 1000.0)
    plt.show()

def show_debug_detection(img_rgb, icons):
    dbg = img_rgb.copy()
    for icon in icons:
        cv2.rectangle(
            dbg,
            (icon.x, icon.y),
            (icon.x + icon.w, icon.y + icon.h),
            (255, 0, 0),
            1,
        )
        cv2.putText(
            dbg,
            NAME_TO_CHAR.get(icon.kind, "?"),
            (icon.x, max(10, icon.y - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    plt.figure(figsize=(10, 10))
    plt.imshow(dbg)
    plt.title("Detected icons")
    plt.axis("off")
    plt.show()

# ============================================================
# MAIN
# ============================================================
def main():
    img_rgb = load_image(IMAGE_PATH)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    step, col_lines, row_lines = infer_grid_step(gray)
    print(f"Inferred grid step: {step}px")

    vertical_walls, horizontal_walls = build_wall_matrices(gray, step, n=MAZE_SIZE)

    icons, color_mask = detect_colored_icons(img_rgb, step)
    obj_matrix = build_object_matrix(icons, n=MAZE_SIZE)

    print("\nDetected icons:")
    for i, icon in enumerate(icons):
        print(
            f"{i:02d}: cell=({icon.row},{icon.col}) "
            f"kind={NAME_TO_CHAR.get(icon.kind, '?')} "
            f"bbox=({icon.x},{icon.y},{icon.w},{icon.h}) "
            f"rgb_mean={tuple(round(v, 1) for v in icon.rgb_mean)}"
        )

    print("\nObject matrix as characters:")
    print(print_symbol_matrix(obj_matrix))

    if SHOW_DEBUG:
        show_debug_detection(img_rgb, icons)

    draw_live_map(vertical_walls, horizontal_walls, obj_matrix, icons, step)

if __name__ == "__main__":
    main()