# MazeBot

MazeBot is a maze-solving project built around two related ideas:

1. Blind exploration:
   The system first explores a maze without full certainty about walls, teleports, or fire timing. It builds up its own discovered map over multiple episodes.
2. RL execution:
   A Q-learning policy then decides how to act turn by turn while following a path toward the goal, especially around confusion and timing-sensitive situations.

The current project has two main runner files:

- `run.py`
  Full two-phase pipeline on `maze_gamma.png`.
  Phase 1 builds blind exploration knowledge.
  Phase 2 uses the learned Q-table plus the discovered route to run the endgame.
- `run_RL.py`
  Pure RL runner on `maze_alpha.png`.
  It trains a Q-table if one is missing, then runs a final visualized episode.

## How The System Works

### Phase 1: Blind Exploration

This phase is used in `run.py`.

The explorer starts with only minimal knowledge:

- start cell
- goal cell
- cells it has visited
- walls it has physically bumped into
- open edges it has successfully crossed
- teleporter mappings it has personally discovered
- fire phases that have already killed it

Over many exploration episodes, the code keeps and reuses discovered knowledge. The result is a partial but useful map of the maze. Once exploration has found enough of the maze, the project extracts a shortest path through the discovered graph.

Important idea:

- exploration is about building knowledge
- it is not using the full ground-truth maze for planning

### Phase 2: RL Endgame

This is also used in `run.py`.

After exploration produces a discovered route, the code builds a route-following agent. That agent uses:

- the discovered maze representation from exploration
- the fixed route found in the discovered graph
- a Q-learning policy loaded from `qtable.json`

The RL policy does not choose arbitrary maze directions directly. Instead, it chooses among higher-level meta actions:

- follow the discovered route normally
- follow the discovered route inverted
- wait

That matters because the maze contains confusion tiles and moving hazards.

### Pure RL Mode

This is what `run_RL.py` does.

It skips blind exploration and works directly with the full parsed maze. If no Q-table exists, it trains one. If a Q-table already exists, it reuses it and runs the final visualized episode.

## High-Level Flow

### `run.py`

1. Parse `maze_beta.png`
2. Run blind exploration across multiple episodes
3. Build a discovered route to the goal
4. Load the existing Q-table from `qtable.json`
5. Build the endgame route-execution agent
6. Write the phase 2 JSON report
7. Visualize the final run

### `run_RL.py`

1. Parse `maze_alpha.png`
2. Load or train a Q-table
3. Run visual training checks every few episodes
4. Run a final visualized episode

## File Tree And Responsibilities

```text
MazeBot/
├── run.py
├── run_RL.py
├── evaluation.py
├── route_execution.py
├── agent.py
├── qlearning.py
├── astar.py
├── visualizer.py
├── qtable.json
├── phase2_report.json
├── maze_alpha.png
├── maze_beta.png
├── environment/
│   ├── __init__.py
│   ├── models.py
│   ├── image_parsing.py
│   ├── fire_patterns.py
│   └── runtime.py
└── exploration/
    ├── __init__.py
    ├── config.py
    ├── knowledge.py
    ├── planning.py
    └── explorer.py
```

### Top-Level Files

#### `run.py`

Main two-phase runner.

Responsible for:

- starting blind exploration
- printing exploration summary stats
- loading the saved Q-table
- writing a JSON report for the phase 2 run
- building the route-execution agent
- launching the final endgame visualization

#### `evaluation.py`

Phase 2 evaluation/reporting helper.

Responsible for:

- running a headless phase 2 endgame episode
- collecting execution counts like turns, actions, deaths, wall hits, and path length
- writing `phase2_report.json`
- keeping evaluation separate from the visual replay

#### `run_RL.py`

Standalone RL runner.

Responsible for:

- loading or training a Q-table
- constructing the standard maze agent
- running periodic visual checks during training
- running the final visualized episode

#### `route_execution.py`

Bridge between exploration and RL execution.

Responsible for:

- converting discovered blind knowledge into maze matrices
- building a route-following agent from exploration results
- overriding replanning so the agent stays aligned with the discovered route

#### `agent.py`

Core RL-controlled maze agent.

Responsible for:

- tracking current position and visited cells
- asking for the next step along the current path
- converting a chosen meta action into a primitive environment action
- updating the Q-table from episode feedback

#### `qlearning.py`

Q-learning implementation.

Responsible for:

- defining RL state and meta actions
- epsilon-greedy action selection
- temporal-difference updates
- reward calculation
- Q-table save/load

#### `astar.py`

Generic A* search implementation.

Responsible for:

- shortest-path search over a neighbor function
- returning either the basic path or the debug-rich search result

#### `visualizer.py`

Matplotlib animation layer.

Responsible for:

- drawing the maze
- showing walls, hazards, path, and agent position
- stepping through the episode one action at a time

#### `qtable.json`

Saved Q-learning table.

Responsible for:

- storing learned Q-values
- storing the latest epsilon value that was saved

### `environment/` Package

This package contains the full environment implementation and was split so the old monolithic environment logic is easier to understand.

#### `environment/models.py`

Shared environment definitions.

Responsible for:

- tile constants
- action enum
- turn-result dataclass
- icon dataclass
- color tables

#### `environment/image_parsing.py`

Maze parsing from image files.

Responsible for:

- loading images
- inferring the grid
- detecting walls
- detecting colored icons
- building object matrices
- building teleporter pairs

#### `environment/fire_patterns.py`

Fire-shape processing.

Responsible for:

- extracting fire cells from the maze image
- separating fire into connected components
- estimating fire roots
- rotating fire components into phase sets

#### `environment/runtime.py`

Actual runtime environment.

Responsible for:

- holding parsed maze state
- resetting episodes
- applying movement rules
- applying confusion
- applying teleport effects
- applying fire deaths
- stepping 1 to 5 actions per turn

#### `environment/__init__.py`

Package export layer.

Responsible for:

- re-exporting the commonly used environment symbols so other files can keep importing from `environment`

### `exploration/` Package

This package contains the blind exploration phase used by `run.py`.

#### `exploration/config.py`

Exploration-only constants.

Responsible for:

- maze size
- fire timing constants
- training budgets
- logging cadence
- early-stop thresholds

#### `exploration/knowledge.py`

Explorer memory model.

Responsible for:

- storing visited cells
- storing discovered open and blocked edges
- storing discovered teleports
- storing known dangerous fire phases
- providing helper functions for adjacency and fire timing

#### `exploration/planning.py`

Exploration planning logic.

Responsible for:

- greedy frontier probe selection
- shortest path through the discovered graph
- timed planning that accounts for fire phases

#### `exploration/explorer.py`

Blind exploration engine.

Responsible for:

- physically stepping through the environment
- updating discovered knowledge after each move
- handling deaths, teleports, waits, and wall bumps
- running repeated exploration episodes
- returning the best accumulated knowledge

#### `exploration/__init__.py`

Small export layer.

Responsible for:

- exposing the specific exploration pieces used by `run.py`

## Important Maze Concepts

### Walls

Walls are inferred from dark grid lines in the source image and stored as vertical and horizontal wall matrices.

### Confusion

Confusion reverses directional controls for a short time. This is why the RL policy has both:

- normal follow action
- inverted follow action

### Teleporters

Colored teleporter cells are paired together. Entering one instantly changes the agent’s position.

### Fire

Fire is dynamic. The environment constructs phase sets by rotating detected fire geometry, then advances those phases over time.

## How To Run

### Run The Full Two-Phase Pipeline

```bash
python3 run.py
```

Notes:

- uses `maze_gamma.png`
- first performs blind exploration
- loads `qtable.json`
- then performs RL-guided route execution

### Run The Pure RL Pipeline

```bash
python3 run_RL.py
```

Notes:

- uses `maze_alpha.png`
- trains a Q-table if one does not already exist
- then runs a final visualized episode

## Dependencies

The code currently relies on:

- `numpy`
- `matplotlib`
- `opencv-python` or another Python package that provides `cv2`

If `cv2` is missing, the environment package cannot load maze images.

## Summary

The project is organized around two layers of intelligence:

- exploration discovers a usable map
- RL decides how to follow the current path safely and robustly inside the maze dynamics

`run.py` combines both phases.
`run_RL.py` is the direct RL-only path.

## AI USE 
- chatgpt for brainstorming ideas and different approaches and Syntax
