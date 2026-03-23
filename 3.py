import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass

from agent import MazeAgent

# ============================================================
# CONFIG
# ============================================================
IMAGE_PATH = "maze_5.png"
FIRE_SOURCE_IMAGE = "maze_5.png"
SHOW_DEBUG = True
MAZE_SIZE = 64
FRAME_MS = 1
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

TARGET_COLORS = {
    FIRE:       (255, 145, 76),   # #ff914c
    CONFUSION:  (255, 222, 89),   # #ffde59
    TP_PURPLE:  (140, 82, 255),   # #8c52ff
    TP_RED:     (255, 49, 50),    # #ff3132
    TP_GREEN:   (1, 191, 99),     # #01bf63
    START:      (15, 192, 223),   # #0fc0df
    GOAL:       (0, 74, 173),     # #004aad
}

DISPLAY_COLORS = {
    EMPTY:      np.array([1.00, 1.00, 1.00]),
    FIRE:       np.array([255, 145, 76]) / 255.0,
    CONFUSION:  np.array([255, 222, 89]) / 255.0,
    TP_PURPLE:  np.array([140, 82, 255]) / 255.0,
    TP_RED:     np.array([255, 49, 50]) / 255.0,
    TP_GREEN:   np.array([1, 191, 99]) / 255.0,
    START:      np.array([15, 192, 223]) / 255.0,
    GOAL:       np.array([0, 74, 173]) / 255.0,
    UNKNOWN:    np.array([0.75, 0.75, 0.75]),
}

COL_AGENT   = np.array([1.00, 0.15, 0.15])
COL_VISITED = np.array([0.60, 0.82, 1.00])
COL_PATH    = np.array([1.00, 0.95, 0.55])

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
# HELPERS
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

def detect_colored_icons(img_rgb, step):
    maxc = img_rgb.max(axis=2)
    minc = img_rgb.min(axis=2)
    sat = maxc - minc

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

def build_object_matrix(icons, n=64):
    obj = np.zeros((n, n), dtype=np.int32)
    for icon in icons:
        obj[icon.row, icon.col] = icon.kind
    return obj

def print_symbol_matrix(obj):
    lines = []
    for r in range(obj.shape[0]):
        lines.append("".join(NAME_TO_CHAR.get(v, "?") for v in obj[r]))
    return "\n".join(lines)

def find_single_cell(obj_matrix, target_value, name):
    cells = list(zip(*np.where(obj_matrix == target_value)))
    if len(cells) != 1:
        raise ValueError(f"Expected exactly 1 {name}, found {len(cells)}")
    return cells[0]

def build_teleport_pairs(obj_matrix):
    teleport_pairs = {}
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

def extract_fire_cells_from_image(path, step):
    img_rgb = load_image(path)
    icons, _ = detect_colored_icons(img_rgb, step)
    obj = build_object_matrix(icons, n=MAZE_SIZE)
    return set(zip(*np.where(obj == FIRE)))


def split_fire_components(cells):
    cells = set(cells)
    components = []
    seen = set()

    for start in cells:
        if start in seen:
            continue

        stack = [start]
        comp = set()
        seen.add(start)

        while stack:
            r, c = stack.pop()
            comp.add((r, c))

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nb = (r + dr, c + dc)
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)

        components.append(comp)

    return components


def find_fire_root(component):
    comp = set(component)
    candidates = []

    for r, c in comp:
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in comp:
                    neighbors.append((dr, dc))

        if len(neighbors) == 2:
            v1, v2 = neighbors
            # Root of the V has two branch vectors that are not exact opposites.
            if not (v1[0] == -v2[0] and v1[1] == -v2[1]):
                candidates.append((r, c))

    if len(candidates) == 1:
        return candidates[0]

    # Fallback: choose cell nearest to component centroid.
    cr = sum(r for r, _ in comp) / len(comp)
    cc = sum(c for _, c in comp) / len(comp)
    return min(comp, key=lambda p: (p[0] - cr) ** 2 + (p[1] - cc) ** 2)


def rotate_point_about_root_90_clockwise(point, root):
    pr, pc = point
    rr, rc = root
    dr = pr - rr
    dc = pc - rc
    return (rr + dc, rc - dr)


def rotate_component_about_root(component, root, quarter_turns, n):
    out = set(component)
    turns = quarter_turns % 4

    for _ in range(turns):
        out = {rotate_point_about_root_90_clockwise(p, root) for p in out}

    return {(r, c) for r, c in out if 0 <= r < n and 0 <= c < n}


def build_rotating_fire_phase_sets(base_fire_cells, n):
    components = split_fire_components(base_fire_cells)
    roots = [find_fire_root(comp) for comp in components]

    phases = []
    for k in range(4):
        phase_cells = set()
        for comp, root in zip(components, roots):
            phase_cells |= rotate_component_about_root(comp, root, k, n)
        phases.append(phase_cells)

    return phases

def build_display(obj_matrix, agent, start, goal, fire_phase_sets):
    n = obj_matrix.shape[0]
    disp = np.ones((n, n, 3), dtype=float)

    for r in range(n):
        for c in range(n):
            tile = obj_matrix[r, c]
            if tile != EMPTY and tile != FIRE:
                disp[r, c] = DISPLAY_COLORS.get(tile, DISPLAY_COLORS[UNKNOWN])

    active_fire = agent.get_active_fire_cells()
    for r, c in active_fire:
        disp[r, c] = DISPLAY_COLORS[FIRE]

    for cell in agent.get_remaining_path()[1:]:
        r, c = cell
        disp[r, c] = disp[r, c] * 0.45 + COL_PATH * 0.55

    if getattr(agent, "visited_mask", None) is not None:
        visited_rows, visited_cols = np.where(agent.visited_mask)
        for r, c in zip(visited_rows, visited_cols):
            disp[r, c] = disp[r, c] * 0.35 + COL_VISITED * 0.65
    else:
        for cell in agent.visited[:-1]:
            r, c = cell
            disp[r, c] = disp[r, c] * 0.35 + COL_VISITED * 0.65

    sr, sc = start
    gr, gc = goal
    disp[sr, sc] = DISPLAY_COLORS[START]
    disp[gr, gc] = DISPLAY_COLORS[GOAL]

    ar, ac = agent.position
    disp[ar, ac] = COL_AGENT
    return disp

def draw_static_walls(ax, vertical_walls, horizontal_walls, n):
    for r in range(n):
        for c in range(n + 1):
            if vertical_walls[r, c]:
                ax.plot([c, c], [r, r + 1], color="black", linewidth=1)

    for r in range(n + 1):
        for c in range(n):
            if horizontal_walls[r, c]:
                ax.plot([c, c + 1], [r, r], color="black", linewidth=1)

def draw_marker_labels(ax, obj_matrix):
    n = obj_matrix.shape[0]
    for r in range(n):
        for c in range(n):
            tile = obj_matrix[r, c]
            if tile != EMPTY and tile != FIRE:
                ax.text(c + 0.5, r + 0.56, NAME_TO_CHAR.get(tile, "?"),
                        ha="center", va="center", fontsize=7)

def show_debug_detection(img_rgb, icons):
    dbg = img_rgb.copy()
    for icon in icons:
        cv2.rectangle(dbg, (icon.x, icon.y), (icon.x + icon.w, icon.y + icon.h), (255, 0, 0), 1)
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
if __name__ == "__main__":
    print("Loading maze...")
    img_rgb = load_image(IMAGE_PATH)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    step, col_lines, row_lines = infer_grid_step(gray)
    print(f"Inferred grid step: {step}px")

    vertical_walls, horizontal_walls = build_wall_matrices(gray, step, n=MAZE_SIZE)

    icons, color_mask = detect_colored_icons(img_rgb, step)
    obj_matrix = build_object_matrix(icons, n=MAZE_SIZE)

    base_fire_cells = extract_fire_cells_from_image(FIRE_SOURCE_IMAGE, step)
    fire_phase_sets = build_rotating_fire_phase_sets(base_fire_cells, MAZE_SIZE)

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

    start = find_single_cell(obj_matrix, START, "start")
    goal = find_single_cell(obj_matrix, GOAL, "goal")
    teleport_pairs = build_teleport_pairs(obj_matrix)

    print(f"\nStart: {start}")
    print(f"Goal : {goal}")

    print("\nTeleport pairs:")
    if teleport_pairs:
        shown = set()
        for a, b in teleport_pairs.items():
            if (b, a) not in shown:
                print(f"  {a} <-> {b}")
                shown.add((a, b))
    else:
        print("  none")

    print("\nFire phase counts:")
    for i, s in enumerate(fire_phase_sets):
        print(f"  phase {i}: {len(s)} cells")

    if SHOW_DEBUG:
        show_debug_detection(img_rgb, icons)

    agent = MazeAgent(
        start=start,
        goal=goal,
        vertical_walls=vertical_walls,
        horizontal_walls=horizontal_walls,
        obj_matrix=obj_matrix,
        teleport_pairs=teleport_pairs,
        fire_phase_sets=fire_phase_sets,
    )
    agent.visited_mask = np.zeros((MAZE_SIZE, MAZE_SIZE), dtype=bool)
    agent.mark_visited(start)

    ok = agent.find_initial_path()
    if not ok:
        print("No path found from start to goal.")
        raise SystemExit

    print(f"\nInitial A* path length: {len(agent.path)}")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, MAZE_SIZE)
    ax.set_ylim(MAZE_SIZE, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    im = ax.imshow(
        build_display(obj_matrix, agent, start, goal, fire_phase_sets),
        extent=(0, MAZE_SIZE, MAZE_SIZE, 0),
        interpolation="nearest"
    )

    draw_static_walls(ax, vertical_walls, horizontal_walls, MAZE_SIZE)
    draw_marker_labels(ax, obj_matrix)

    title = ax.set_title("Step 0 | Fire: 0°", fontsize=10)

    ani_holder = {"ani": None}

    def update(_frame):
        event = agent.step()
        im.set_data(build_display(obj_matrix, agent, start, goal, fire_phase_sets))

        phase = (agent.total_steps // 5) % 4
        phase_text = {0: "0°", 1: "90°", 2: "180°", 3: "270°"}[phase]

        if event == "goal":
            title.set_text(f"GOAL REACHED | Steps: {agent.total_steps} | Fire: {phase_text}")
            title.set_color("green")
            if ani_holder["ani"] is not None:
                ani_holder["ani"].event_source.stop()

        elif event == "dead":
            title.set_text(
                f"DEAD -> RESPAWN NEXT | Steps: {agent.total_steps} | Fire: {phase_text} | Deaths: {agent.death_count}"
            )
            title.set_color("red")

        elif event == "respawn":
            title.set_text(
                f"RESPAWN | Steps: {agent.total_steps} | Fire: {phase_text} | Deaths: {agent.death_count} | Known pits: {len(agent.known_pits)}"
            )
            title.set_color("darkorange")

        elif event == "stuck":
            title.set_text(
                f"STUCK (NO PATH) | Steps: {agent.total_steps} | Deaths: {agent.death_count} | Known pits: {len(agent.known_pits)}"
            )
            title.set_color("dimgray")
            if ani_holder["ani"] is not None:
                ani_holder["ani"].event_source.stop()

        elif event == "wait":
            title.set_text(
                f"WAITING FOR FIRE PHASE | Steps: {agent.total_steps} | Fire: {phase_text} | Deaths: {agent.death_count}"
            )
            title.set_color("saddlebrown")

        elif event == "teleport":
            title.set_text(f"TELEPORT | Steps: {agent.total_steps} | Fire: {phase_text} | Replans: {agent.replans}")
            title.set_color("purple")

        else:
            title.set_text(f"Step {agent.total_steps} | Fire: {phase_text} | Replans: {agent.replans}")
            title.set_color("black")

        return [im, title]

    ani_holder["ani"] = animation.FuncAnimation(
        fig,
        update,
        frames=5000,
        interval=FRAME_MS,
        blit=False,
        repeat=False
    )

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)
    print(f"Start       : {start}")
    print(f"Goal        : {goal}")
    print(f"Steps       : {agent.total_steps}")
    print(f"Replans     : {agent.replans}")
    print(f"Deaths      : {agent.death_count}")
    print(f"Confusions  : {agent.confusion_count}")
    print(f"Reached goal: {agent.done}")
    print(f"Stuck       : {agent.failed}")
    print(f"Dead        : {agent.dead}")
    print("=" * 50)