import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation

from agent import MazeAgent, HazardMap

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# Usage:
#   python 3.py          — normal mode, agent detects and avoids hazards
#   python 3.py --blind  — blind mode, agent walks straight through hazards
# ═══════════════════════════════════════════════════════════════════════
IMAGE_PATH = '/Users/amrindersingh/Desktop/AI_Porject/Group Project Preliminary Specifications/MAZE_0.png'
N_PITS = 4
N_TELEPORTS = 3
VISION_RANGE = 30
STEPS_PER_FRAME = 1
STRIDE = 10
FRAME_MS = 80
RANDOM_SEED = None

AGENT_RADIUS = 8
GOAL_RADIUS = 6
HAZARD_RADIUS = 10
VISION_ALPHA = 0.45

HAZARD_DETECTION = '--blind' not in sys.argv

# ── Colors ─────────────────────────────────────────────────────────────
COL_WALL         = np.array([0.08, 0.08, 0.12])
COL_PATH         = np.array([0.93, 0.93, 0.90])
COL_VISITED      = np.array([0.35, 0.65, 1.00])
COL_AGENT        = np.array([1.00, 0.12, 0.12])
COL_GOAL         = np.array([0.00, 0.92, 0.25])
COL_PIT          = np.array([1.00, 0.25, 0.00])
COL_TELEPORT     = np.array([0.70, 0.00, 1.00])
COL_KNOWN_HAZARD = np.array([0.80, 0.80, 0.00])
COL_ASTAR        = np.array([1.00, 0.90, 0.05])
COL_VISION       = np.array([0.90, 0.90, 0.20])
COL_FLASH        = np.array([1.00, 0.10, 0.80])
COL_EXPLORED     = np.array([0.20, 0.80, 0.60])
COL_TELEPORT_FLASH = np.array([0.50, 0.00, 1.00])


# ═══════════════════════════════════════════════════════════════════════
# MAZE UTILS
# ═══════════════════════════════════════════════════════════════════════
def maze_to_matrix(image_path, threshold=127):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return (binary > threshold).astype(int)


def find_border_openings(matrix):
    rows, cols = matrix.shape
    openings = []
    for c in range(cols):
        if matrix[0, c] == 1:
            openings.append((0, c))
    for c in range(cols):
        if matrix[rows - 1, c] == 1:
            openings.append((rows - 1, c))
    for r in range(rows):
        if matrix[r, 0] == 1:
            openings.append((r, 0))
    for r in range(rows):
        if matrix[r, cols - 1] == 1:
            openings.append((r, cols - 1))

    seen, unique = set(), []
    for p in openings:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    if len(unique) < 2:
        raise ValueError("Need ≥2 border openings")
    return unique[0], unique[-1]


def get_open_cells(matrix):
    rows, cols = matrix.shape
    return [
        (r, c)
        for r in range(1, rows - 1)
        for c in range(1, cols - 1)
        if matrix[r, c] == 1
    ]


# ═══════════════════════════════════════════════════════════════════════
# DRAW HELPERS
# ═══════════════════════════════════════════════════════════════════════
def paint_circle(display, center, radius, color, rows, cols, matrix=None):
    r0, c0 = center
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr * dr + dc * dc <= radius * radius:
                nr, nc = r0 + dr, c0 + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if matrix is None or matrix[nr, nc] == 1:
                        display[nr, nc] = color


def build_display(matrix, agent, hazard_map, goal, flash=False, teleport_flash=False):
    rows, cols = matrix.shape
    display = np.zeros((rows, cols, 3), dtype=float)

    for r in range(rows):
        for c in range(cols):
            display[r, c] = COL_PATH if matrix[r, c] == 1 else COL_WALL

    EXPLORED_MAX_FRAMES = 18
    for cell, frames_left in agent.explored_fade.items():
        r, c = cell
        if 0 <= r < rows and 0 <= c < cols and matrix[r, c] == 1:
            alpha = 0.10 + 0.45 * (frames_left / EXPLORED_MAX_FRAMES)
            display[r, c] = display[r, c] * (1 - alpha) + COL_EXPLORED * alpha

    for cell in agent.visited[:-1]:
        r, c = cell
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and matrix[nr, nc] == 1:
                    display[nr, nc] = COL_VISITED

    upcoming = agent.get_remaining_path()
    vision_cells = upcoming[:VISION_RANGE]
    for cell in vision_cells:
        r, c = cell
        display[r, c] = display[r, c] * (1 - VISION_ALPHA) + COL_VISION * VISION_ALPHA

    for i, cell in enumerate(upcoming[VISION_RANGE:]):
        if i % 3 == 0:
            paint_circle(display, cell, 2, COL_ASTAR, rows, cols)

    for pos in hazard_map.pit_positions:
        paint_circle(display, pos, HAZARD_RADIUS, COL_PIT, rows, cols, matrix)
    for pos in hazard_map.teleport_positions:
        paint_circle(display, pos, HAZARD_RADIUS, COL_TELEPORT, rows, cols, matrix)

    for pos in agent.known_hazards:
        r0, c0 = pos
        for dr in range(-(HAZARD_RADIUS + 3), HAZARD_RADIUS + 4):
            for dc in range(-(HAZARD_RADIUS + 3), HAZARD_RADIUS + 4):
                dist = dr * dr + dc * dc
                inner = (HAZARD_RADIUS + 1) ** 2
                outer = (HAZARD_RADIUS + 3) ** 2
                if inner < dist <= outer:
                    nr, nc = r0 + dr, c0 + dc
                    if 0 <= nr < rows and 0 <= nc < cols and matrix[nr, nc] == 1:
                        display[nr, nc] = COL_KNOWN_HAZARD

    paint_circle(display, goal, GOAL_RADIUS, COL_GOAL, rows, cols)
    paint_circle(display, goal, GOAL_RADIUS + 2, np.array([1.0, 1.0, 1.0]), rows, cols, matrix)

    if flash:
        paint_circle(display, agent.get_position(), AGENT_RADIUS + 10, COL_FLASH, rows, cols, matrix)

    if teleport_flash:
        paint_circle(display, agent.get_position(), AGENT_RADIUS + 10, COL_TELEPORT_FLASH, rows, cols, matrix)

    paint_circle(display, agent.get_position(), AGENT_RADIUS, COL_AGENT, rows, cols)
    paint_circle(display, agent.get_position(), 2, np.array([1.0, 1.0, 1.0]), rows, cols)

    return display


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Loading maze...")
    maze = maze_to_matrix(IMAGE_PATH)
    print(f"  Shape: {maze.shape}")

    start, goal = find_border_openings(maze)
    print(f"  Start : {start}  |  Goal : {goal}")

    open_cells = get_open_cells(maze)
    print(f"  Open cells: {len(open_cells)}")

    print(f"\nPlacing {N_PITS} death pits + {N_TELEPORTS} teleport pads...")
    hazard_map = HazardMap(N_PITS, N_TELEPORTS, open_cells, seed=RANDOM_SEED)
    print(f"  Pits      : {sorted(hazard_map.pit_positions)}")
    print(f"  Teleports : {sorted(hazard_map.teleport_positions)}")

    print(f"\nRunning initial A* (hazard-blind)...")
    agent = MazeAgent(
        start=start,
        goal=goal,
        vision_range=VISION_RANGE,
        stride=STRIDE,
        hazard_radius=HAZARD_RADIUS
    )
    agent.hazard_detection = HAZARD_DETECTION
    print(f"  Hazard detection: {'ON' if HAZARD_DETECTION else 'OFF (blind mode)'}")

    if not agent.find_path(maze, hazard_map):
        print("No path found — exiting.")
        exit()

    # In blind mode force hazards onto the path so consequences are visible
    if not HAZARD_DETECTION:
        path_cells = agent.path[50:-50]
        hazard_map.force_on_path(path_cells, N_PITS, N_TELEPORTS)
        print(f"  Blind mode: hazards forced onto path")
        print(f"  Pits      : {sorted(hazard_map.pit_positions)}")
        print(f"  Teleports : {sorted(hazard_map.teleport_positions)}")

    print("\nAnimating...\n")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis("off")
    fig.patch.set_facecolor('#0d0d0d')

    init_disp = build_display(maze, agent, hazard_map, goal)
    im = ax.imshow(init_disp, interpolation='nearest')

    mode_label = "  [BLIND MODE]" if not HAZARD_DETECTION else ""
    title = ax.set_title(
        f"Step 0  |  A* path: {len(agent.path)} cells  |  "
        f"Vision: {VISION_RANGE}  |  Stride: {STRIDE}{mode_label}",
        color='orange' if not HAZARD_DETECTION else 'white',
        fontsize=10, pad=8
    )

    legend_elements = [
        mpatches.Patch(color=COL_AGENT,         label='Agent'),
        mpatches.Patch(color=COL_GOAL,          label='Goal'),
        mpatches.Patch(color=COL_PIT,           label='Death pit'),
        mpatches.Patch(color=COL_TELEPORT,      label='Teleport pad'),
        mpatches.Patch(color=COL_KNOWN_HAZARD,  label='Known hazard ring'),
        mpatches.Patch(color=COL_VISION,        label=f'Vision cone ({VISION_RANGE} cells)'),
        mpatches.Patch(color=COL_ASTAR,         label='A* planned path'),
        mpatches.Patch(color=COL_VISITED,       label='Visited trail'),
        mpatches.Patch(color=COL_EXPLORED,      label='A* explored (search frontier)'),
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=7,
        framealpha=0.85,
        facecolor='#1a1a1a',
        labelcolor='white'
    )

    state = {'flash': False, 'flash_frames': 0, 'teleport_flash': False}

    def update(frame):
        if agent.is_done():
            disp = build_display(maze, agent, hazard_map, goal)
            im.set_data(disp)
            title.set_text(
                f"GOAL REACHED!  Steps: {agent.total_steps}  |  "
                f"Replans: {agent.replan_count}  |  "
                f"Hazards learned: {len(agent.known_hazards)}"
            )
            title.set_color('lime')
            return [im, title]

        if agent.stuck:
            disp = build_display(maze, agent, hazard_map, goal)
            im.set_data(disp)
            title.set_text(f"DEAD — fell into pit at step {agent.total_steps}")
            title.set_color('red')
            return [im, title]

        if state['flash_frames'] > 0:
            state['flash_frames'] -= 1
            disp = build_display(
                maze, agent, hazard_map, goal,
                flash=state['flash'],
                teleport_flash=state['teleport_flash']
            )
            im.set_data(disp)
            return [im, title]

        results = agent.take_steps(n=STEPS_PER_FRAME)
        spotted  = any(e == 'spotted'   for _, e in results)
        pit      = any(e == 'pit'       for _, e in results)
        teleport = any(e == 'teleport'  for _, e in results)

        if pit:
            disp = build_display(maze, agent, hazard_map, goal, flash=True)
            im.set_data(disp)
            title.set_text(f"DEAD — fell into pit at step {agent.total_steps}")
            title.set_color('red')
            return [im, title]

        if teleport:
            state['teleport_flash'] = True
            state['flash'] = False
            state['flash_frames'] = 6
        elif spotted:
            state['flash'] = True
            state['teleport_flash'] = False
            state['flash_frames'] = 3
        else:
            state['flash'] = False
            state['teleport_flash'] = False

        disp = build_display(maze, agent, hazard_map, goal,
                             flash=spotted,
                             teleport_flash=teleport)
        im.set_data(disp)

        if teleport:
            status = "  TELEPORTED — replanning!"
        elif spotted:
            status = "  HAZARD SPOTTED — replanning!"
        else:
            status = ""

        mode = "  [BLIND MODE]" if not HAZARD_DETECTION else ""
        title.set_color('orange' if not HAZARD_DETECTION else 'white')
        title.set_text(
            f"Step {agent.total_steps}  |  "
            f"Remaining: {len(agent.get_remaining_path())}  |  "
            f"Replans: {agent.replan_count}  |  "
            f"Known hazards: {len(agent.known_hazards)}"
            f"{status}{mode}"
        )
        return [im, title]

    total_frames = (len(agent.path) // max(1, STRIDE)) * 3 + 100
    ani = animation.FuncAnimation(
        fig, update,
        frames=total_frames,
        interval=FRAME_MS,
        blit=False,
        repeat=False
    )

    plt.tight_layout()
    plt.show()

    print("\n" + "═" * 50)
    print("  FINAL SUMMARY")
    print("═" * 50)
    print(f"  Mode            : {'BLIND' if not HAZARD_DETECTION else 'NORMAL'}")
    print(f"  Total steps     : {agent.total_steps}")
    print(f"  Replans         : {agent.replan_count}")
    print(f"  Hazards learned : {len(agent.known_hazards)}")
    print("═" * 50)