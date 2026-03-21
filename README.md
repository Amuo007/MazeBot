# 🧩 Maze Solver — PNG to Animated Agent

A computer-vision + pathfinding pipeline that takes a **painted maze image** (PNG),
automatically reads its walls, colored tiles, and special icons, then animates an AI
agent solving it in real time — complete with teleporters, rotating fire hazards, and
confusion tiles.

---

## 📁 Project Files

| File | Role |
|------|------|
| `1.py` | **Inspector** — loads a maze PNG, detects icons, prints the grid as text, shows a static matplotlib map |
| `3.py` | **Animator** — full simulation with live animation, fire rotation, and the agent walking the maze |
| `agent.py` | **Agent brain** — A\* pathfinding, teleport handling, fire avoidance, and step-by-step movement logic |
| `maze_5.png` | Base maze image (phase 0 / fire at 0°) |
| `2.png` / `3.png` / `4.png` | Same maze with fire tiles rotated 90° / 180° / 270° |

> **Images** — The solver expects `maze_5.png` as the main maze and `2.png`, `3.png`, `4.png`
> as the three rotated fire-phase variants. All four should sit in the same folder as the scripts.
> Example of what these look like: a white-background 64×64 grid with black walls drawn as lines
> and colored blobs painted inside cells to mark special tiles.

---

## 🧱 Tile Types

Each cell in the maze can be one of these:

| Symbol | Color (hex) | Meaning |
|--------|-------------|---------|
| `.` | white | Empty — safe to walk |
| `F` | `#ff914c` orange | **Fire** — kills agent on contact |
| `C` | `#ffde59` yellow | **Confusion** — (reserved for future penalty) |
| `P` | `#8c52ff` purple | **Teleporter** — purple pair |
| `R` | `#ff3132` red | **Teleporter** — red pair |
| `G` | `#01bf63` green | **Teleporter** — green pair |
| `S` | `#0fc0df` cyan | **Start** — agent spawns here |
| `E` | `#004aad` blue | **Goal** — agent must reach this |

Teleporters come in pairs (or cycles): stepping on one instantly moves the agent to its partner.

---

## 🔄 How the Fire Rotates

Fire isn't static. Every **5 steps** the active fire set advances through a 4-phase cycle
(0° → 90° → 180° → 270°), each loaded from a separate image file:

```
Phase 0 (steps 0–4)   → maze_5.png
Phase 1 (steps 5–9)   → 2.png
Phase 2 (steps 10–14) → 3.png
Phase 3 (steps 15–19) → 4.png
Phase 4 (steps 20–24) → maze_5.png  ← wraps back
```

This means a cell that's safe now may be lethal two steps later — the agent must replan
around fire dynamically.

---

## 🖼️ Step 1 — PNG → Grid Matrix

### 1a. Detect the grid lines

`infer_grid_step(gray)` converts the image to grayscale and looks for columns/rows
where more than 25% of pixels are near-black (< 40 brightness). These are the wall lines.
Closely spaced black pixels are merged into single "line" positions, and the median gap
between consecutive lines becomes the **cell step size** (in pixels).

```
colsum  = how many black pixels per column
col_peaks = columns where colsum is large  →  these are wall lines
step_x  = median gap between wall line positions
step = average of step_x and step_y
```

### 1b. Build wall matrices

`build_wall_matrices()` creates two boolean matrices:

- `vertical_walls[r, c]` — is there a wall on the **left** edge of cell (r, c)?
- `horizontal_walls[r, c]` — is there a wall on the **top** edge of cell (r, c)?

For each expected wall position, it samples a thin slice of the grayscale image and
checks if more than 35% of that slice is dark. If yes → wall present.

### 1c. Detect colored tiles

`detect_colored_icons()` works in pure color space:

1. Compute **saturation** = `max(R,G,B) - min(R,G,B)` for every pixel.
2. Mask pixels where saturation > 40 AND brightness > 60 (colorful, non-black, non-white).
3. Morphological open + close to remove noise and fill small gaps.
4. `cv2.connectedComponentsWithStats` groups neighboring colored pixels into blobs.
5. Each blob is filtered: too small (< 8px) or too large (> 2 cells wide) → discarded.
6. The mean RGB of the blob is computed, then `classify_icon()` finds the nearest known
   color using **Euclidean distance in RGB space** (tolerance = 45 units).
7. The blob centroid is converted to `(row, col)` by dividing by the step size.

This gives an `obj_matrix[64][64]` where each cell is 0 (empty) or a tile type code.

---

## 🤖 Step 2 — The Agent (`agent.py`)

### A\* Pathfinding

The agent uses classic **A\*** with Manhattan distance as the heuristic.
The key constraint is `can_move(a, b)` — a move is only valid if there is **no wall**
between the two adjacent cells. It checks `vertical_walls` and `horizontal_walls`
directly, so the path is always physically valid.

```python
# moving up from (r, c) to (r-1, c):
allowed = horizontal_walls[r, c] == 0

# moving right from (r, c) to (r, c+1):
allowed = vertical_walls[r, c+1] == 0
```

### Step loop (`agent.step()`)

Each call to `step()` does one thing:

1. Advance one cell along the current planned path.
2. Check if the new cell is **on fire** → die.
3. Check if the new cell is a **teleporter** → jump to partner, replan from there.
4. Check if the new cell is the **goal** → done.
5. Otherwise → return `"move"` and keep going.

The agent does **not** proactively replan around future fire — it only replans after a
teleport. If it steps into a newly active fire cell, it dies and the simulation ends.

### Teleport handling

When the agent lands on a teleporter cell, it is instantly moved to the partner cell
(looked up from the `teleport_pairs` dict built at startup). A\* is then called again
from the new position to find a fresh path to the goal.

---

## 🎬 Step 3 — Animation (`3.py`)

`matplotlib.animation.FuncAnimation` calls `agent.step()` every `FRAME_MS` milliseconds.
After each step, `build_display()` repaints the 64×64 grid as an RGB image:

- **White** — empty
- **Tile color** — special tiles (start, goal, teleporters, etc.)
- **Orange** — active fire cells (from the current phase)
- **Light blue** — cells the agent has already visited
- **Yellow tint** — cells on the agent's remaining planned path
- **Red** — the agent's current position

Static wall lines are drawn once with `matplotlib.lines` on top of the image,
so they don't flicker.

The title bar updates every frame with the current step count, fire phase, and replan count.
Animation stops automatically when the agent reaches the goal or dies.

---

## ▶️ Running It

**Inspect only (no animation):**
```bash
python 1.py
```
Prints detected icons and the text symbol matrix, shows a static matplotlib map.

**Full animated simulation:**
```bash
python 3.py
```
Opens the live matplotlib window with the agent walking the maze.

**Dependencies:**
```bash
pip install opencv-python numpy matplotlib
```

---

## ⚙️ Config (top of `3.py` / `1.py`)

| Variable | Default | Meaning |
|----------|---------|---------|
| `IMAGE_PATH` | `"maze_5.png"` | Main maze image |
| `FIRE_PHASE_IMAGES` | `[maze_5, 2, 3, 4]` | Four fire rotation images |
| `MAZE_SIZE` | `64` | Grid dimensions (N×N) |
| `FRAME_MS` | `90` | Animation speed in ms per step |
| `COLOR_TOL` | `45` | Max RGB distance to classify a tile color |
| `SHOW_DEBUG` | `True` | Show bounding-box overlay on detected icons |

---

## 📌 Known Limitations

- Agent does **not** predict fire — it will die if the fire phase rotates into its current path mid-walk. No lookahead replanning is implemented.
- Confusion tiles (`C`) are detected and labeled but have no gameplay effect yet.
- The grid must be exactly 64×64 cells for the wall matrices to be sized correctly.
