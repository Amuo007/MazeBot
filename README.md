# 🧭 Silent Cartographer — Maze Navigation Agent

A Python-based maze-solving AI agent that navigates a visually parsed grid maze with dynamic hazards including **fire**, **confusion tiles**, **teleporters**, and **walls**. The agent uses **A\* search** to plan paths and is visualized frame-by-frame using matplotlib animation.

---

## 📁 Project Structure

```
silent-cartographer/
│
├── main.py            # Entry point — wires environment, agent, and visualizer
├── agent.py           # MazeAgent logic: planning, memory, action generation
├── astar.py           # A* search algorithm (with debug info)
├── environment.py     # Maze parser, tile types, physics, and simulation loop
├── visualizer.py      # Matplotlib animator for real-time episode playback
│
└── maze_5.png         # Primary maze image (defines walls, start, goal, tiles)
```

> 💡 **The maze is defined by one image.** Walls are inferred from dark pixels; colored icons define tile types (fire, teleporters, confusion, start, goal). Fire phases are derived by rotating the detected fire geometry in `environment.py`.

PNG image

## 🗺️ How It Works — Pipeline Overview

```
PNG image(s)
     │
     ▼
[environment.py] ── Image parsing
     │                ├─ Infer grid step size (pixel gaps between grid lines)
     │                ├─ Build vertical_walls & horizontal_walls matrices
     │                └─ Detect colored icons → object matrix
     │                └─ fire_phase_sets (four rotated phases from the base fire layout)
     ▼
MazeEnvironment  ── Holds all ground truth
     │                ├─ start, goal positions
     │                ├─ teleport_pairs dict
     │                └─ fire_phase_sets (one set per image)
     │
     ▼
[agent.py]       ── Decision making each turn
     │                ├─ Receives TurnResult (position, dead, confused, etc.)
     │                ├─ Updates AgentMemory (visited, known_safe, etc.)
     │                └─ Calls A* to plan path → converts to list of Actions
     │
     ▼
[astar.py]       ── Pathfinding
     │                └─ Returns path + expanded nodes for visualization
     │
     ▼
[visualizer.py]  ── Renders frame-by-frame animation
                      ├─ Color-codes: agent, path, visited, fire, tiles
                      └─ Draws walls as line overlays
```

---

## 📦 Key Files Explained

### `main.py` — Entry Point

Bootstraps the full episode:
1. Creates `MazeEnvironment` from the base image
2. Creates `MazeAgent` using the environment's parsed data
3. Calls `animate_episode()` which drives the simulation loop

```python
env = MazeEnvironment(image_path="maze_5.png")
agent = MazeAgent(start=env.start, goal=env.goal, ...)
animate_episode(env, agent, max_turns=10000, frame_ms=90)
```

---

### `environment.py` — World Model + Simulation

The most complex file. Handles image parsing AND game physics.

#### Tile Types (constants)

| Constant    | Value | Color (RGB)       | Meaning                        |
|-------------|-------|-------------------|--------------------------------|
| `EMPTY`     | 0     | White             | Passable floor                 |
| `FIRE`      | 1     | Orange `(255,145,76)` | Kills agent on contact     |
| `CONFUSION` | 2     | Yellow `(255,222,89)` | Reverses controls for 2 turns |
| `TP_PURPLE` | 3     | Purple `(140,82,255)` | Teleporter (purple pair)   |
| `TP_RED`    | 4     | Red `(255,49,50)`     | Teleporter (red pair)      |
| `TP_GREEN`  | 5     | Green `(1,191,99)`    | Teleporter (green pair)    |
| `START`     | 6     | Cyan `(15,192,223)`   | Agent spawn point          |
| `GOAL`      | 7     | Blue `(0,74,173)`     | Episode end point          |
| `UNKNOWN`   | 99    | Gray              | Unrecognized tile              |

#### Key Functions

| Function | Purpose |
|---|---|
| `load_image_rgb(path)` | Loads PNG as RGB numpy array |
| `infer_grid_step(gray)` | Finds pixel spacing between grid lines |
| `build_wall_matrices(gray, step, n)` | Returns `vertical_walls[n, n+1]` and `horizontal_walls[n+1, n]` |
| `detect_colored_icons(img_rgb, step)` | Finds colored blobs → assigns to grid cells |
| `build_object_matrix(icons, n)` | Creates `n×n` int matrix of tile types |
| `build_teleport_pairs(obj_matrix)` | Maps each teleporter cell to its partner |
| `extract_fire_cells_from_image(path)` | Extracts fire tile positions from a phase image |

#### `MazeEnvironment` — Simulation Class

| Method | Purpose |
|---|---|
| `reset()` | Puts agent back at start, clears stats |
| `step(actions)` | Executes a list of 1–5 actions, returns `TurnResult` |
| `step_one_action(action, confused)` | Executes a single action, applies tile effects |
| `_apply_tile_effects(result)` | Checks fire, confusion, teleport, goal |
| `apply_confusion(action)` | Reverses direction (UP↔DOWN, LEFT↔RIGHT) |
| `get_active_fire_cells()` | Returns fire positions for current phase |
| `get_episode_stats()` | Returns dict of turns, deaths, cells explored, etc. |

#### Fire Phases

Fire alternates between four rotated phases every 5 actions:
```
phase = (total_actions_executed // 5) % len(fire_phase_sets)
```
This means fire patterns shift over time, requiring the agent to adapt without needing separate phase images.

#### `TurnResult` — What the Agent Receives Back

```python
@dataclass
class TurnResult:
    wall_hits: int          # How many walls were bumped this turn
    current_position: Cell  # (row, col) after the turn
    is_dead: bool           # Stepped into fire → respawned at start
    is_confused: bool       # Under confusion effect this turn
    is_goal_reached: bool   # Reached the goal cell
    teleported: bool        # Used a teleporter
    actions_executed: int   # How many actions ran before stop
```

---

### `agent.py` — Agent Brain

#### `AgentMemory` — What the Agent Remembers

```python
@dataclass
class AgentMemory:
    known_walls: Set[Tuple[Cell, Cell]]   # (a, b) wall pairs discovered
    known_safe: Set[Cell]                 # Cells confirmed non-lethal
    known_pits: Set[Cell]                 # (future use) dangerous cells
    known_confusion: Set[Cell]            # Confusion tiles discovered
    known_teleports: Dict[Cell, Cell]     # Teleporter mappings discovered
    visited: Set[Cell]                    # All cells ever stepped on
```

> 📝 Currently the agent uses **full maze knowledge** (cheats with ground truth walls/tiles). The `AgentMemory` dataclass is the scaffold for a future **fully blind** agent that learns from exploration.

#### `ActionController` — Action Helpers

Converts grid deltas to `Action` enum values:
```python
ActionController.delta_to_action((r1,c1), (r2,c2)) → Action.MOVE_RIGHT
```

#### `MazeAgent` — Core Decision Loop

| Method | Purpose |
|---|---|
| `reset_episode()` | Clears path, position, and memory for a new run |
| `plan_turn(last_result)` | Main method — updates state, runs A*, returns actions |
| `update_from_result(result)` | Applies `TurnResult` to agent's internal position |
| `plan_path(start, goal)` | Runs A* and stores debug data for visualizer |
| `path_to_actions(path, limit)` | Converts cell list → up to 5 `Action` values |
| `can_move(a, b)` | Wall-aware adjacency check |
| `neighbors(cell)` | Returns passable adjacent cells (used as A* neighbor fn) |

---

### `astar.py` — Pathfinding

Standard **A\* search** with Manhattan distance heuristic.

| Function | Purpose |
|---|---|
| `astar_search(start, goal, neighbors_fn)` | Returns the shortest path as `List[Cell]` |
| `astar_search_debug(...)` | Same, plus returns `expanded_order`, `closed_set`, `g_score` for visualization |
| `manhattan(a, b)` | Heuristic: `|Δrow| + |Δcol|` |
| `reconstruct_path(came_from, current)` | Backtracks through parent map to build path |

The debug version returns:
```python
{
  "path": [...],             # Final chosen path
  "expanded_order": [...],   # Nodes popped from heap in order
  "closed_set": {...},       # All nodes fully explored
  "g_score": {...},          # Best cost found to each cell
}
```

This is used by the visualizer to color-code the search frontier.

---

### `visualizer.py` — Matplotlib Animator

Renders a live frame-by-frame animation of the episode.

#### Color Legend

| Color | Meaning |
|---|---|
| 🔴 Red `(1.0, 0.15, 0.15)` | Agent's current position |
| 🔵 Light Blue `COL_VISITED` | Cells the agent has visited |
| 🟡 Yellow `COL_PATH` | Planned path (A* result) |
| 🩵 Pale Blue `COL_SEARCH` | A* expanded nodes (frontier) |
| 🟦 Blue-gray `COL_CLOSED` | A* closed set |
| 🟠 Orange | Active fire tiles |
| 🟣 Purple / Red / Green | Teleporter pairs |
| 🟦 Cyan | Start cell |
| 🔷 Dark Blue | Goal cell |
| ⬜ White | Empty passable floor |

#### Key Functions

| Function | Purpose |
|---|---|
| `animate_episode(env, agent, ...)` | Main animation loop using `FuncAnimation` |
| `build_display(obj_matrix, env, agent)` | Builds `n×n×3` RGB float array for imshow |
| `draw_static_walls(ax, ...)` | Draws wall lines once using matplotlib |
| `draw_marker_labels(ax, obj_matrix)` | Prints S/E/F/C/P/R/G text on tiles |

The animation executes **one action per frame** (not one turn), so you can watch the agent step cell by cell in real time.

---

## 🧩 Key Terms Glossary

| Term | Definition |
|---|---|
| **Cell** | A `(row, col)` tuple — the fundamental grid coordinate unit |
| **Turn** | One planning cycle: agent returns up to 5 actions at once |
| **Action** | Enum: `MOVE_UP`, `MOVE_DOWN`, `MOVE_LEFT`, `MOVE_RIGHT`, `WAIT` |
| **Wall matrix** | 2D array marking whether a boundary between cells is blocked |
| **Fire phase** | Which of the 4 fire-pattern images is currently active (cycles every 5 actions) |
| **Confusion** | Tile effect that reverses all movement for 2 turns |
| **Teleporter** | Stepping on one cell instantly moves agent to its paired cell |
| **TurnResult** | Dataclass returned after each turn with position, death, confusion status, etc. |
| **AgentMemory** | Dataclass storing everything the agent has learned across steps |
| **A\*** | Graph search algorithm using `g(n) + h(n)` to find shortest path |
| **Manhattan distance** | Heuristic `|Δrow| + |Δcol|` — admissible for grid movement |
| **obj_matrix** | `64×64` int array mapping each cell to its tile type constant |

---

## ▶️ Running the Project

```bash
# Install dependencies
pip install numpy opencv-python matplotlib

# Run the visualized episode
python main.py
```

Make sure `maze_5.png` is in the same directory as `main.py`.

---

## 🔭 Future Work

- **Blind exploration mode**: Remove ground-truth wall access; agent discovers walls by bumping into them and stores them in `AgentMemory.known_walls`
- **Fire avoidance**: Route around known fire cells rather than treating them as passable
- **Teleporter learning**: Discover and remember teleport destinations from `TurnResult.teleported`
- **Confusion handling**: Invert planned actions when `is_confused=True` in last result
- **Frontier exploration**: When no known path exists, explore toward unvisited cells first
