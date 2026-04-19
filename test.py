from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from astar import astar_search_debug
from environment import MazeEnvironment
from visualizer import DISPLAY_COLORS, draw_marker_labels, draw_static_walls

Cell = Tuple[int, int]

IMAGE_CANDIDATES = ["test2.PNG", "test2.png"]
ANIMATION_FRAME_MS = 18
MAX_SEARCH_FRAMES = 220
LAST_ANIMATION = None


def resolve_image_path() -> str:
    for candidate in IMAGE_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(f"Could not find any of: {', '.join(IMAGE_CANDIDATES)}")


def teleport_aware_neighbors(env: MazeEnvironment, cell: Cell) -> List[Cell]:
    r, c = cell
    out: List[Cell] = []

    for nb in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
        if not env.can_move(cell, nb):
            continue
        out.append(env.teleport_pairs.get(nb, nb))

    return out


def build_base_display(env: MazeEnvironment) -> np.ndarray:
    n = env.maze_size
    disp = np.ones((n, n, 3), dtype=float)

    for r in range(n):
        for c in range(n):
            tile = int(env.obj_matrix[r, c])
            disp[r, c] = DISPLAY_COLORS.get(tile, DISPLAY_COLORS[0])

    for r, c in env.get_active_fire_cells():
        disp[r, c] = DISPLAY_COLORS[1]

    return disp


def summarize_path(env: MazeEnvironment, path: List[Cell]) -> None:
    teleports_used = []
    for a, b in zip(path, path[1:]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            teleports_used.append((a, b))

    print(f"Start : {env.start}")
    print(f"Goal  : {env.goal}")
    print(f"Path length : {max(0, len(path) - 1)}")
    print(f"Teleports used in path : {len(teleports_used)}")
    for src, dst in teleports_used:
        print(f"  teleport: {src} -> {dst}")


def animate_astar(env: MazeEnvironment, expanded_order: List[Cell], path: List[Cell]) -> None:
    global LAST_ANIMATION

    n = env.maze_size
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.subplots_adjust(right=0.82)

    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    base = build_base_display(env)
    ax.imshow(base, extent=(0, n, n, 0), interpolation="nearest")
    draw_static_walls(ax, env.vertical_walls, env.horizontal_walls, n)
    draw_marker_labels(ax, env.obj_matrix)

    expanded_scatter = ax.scatter([], [], s=18, c="#b7d8f8", marker="s", alpha=0.75)
    path_line, = ax.plot([], [], color="orange", linewidth=2.6)
    agent_dot, = ax.plot([], [], "o", color="#ff3132", markersize=8)
    title = ax.set_title("A* search", fontsize=10)

    if "agg" in plt.get_backend().lower():
        expanded_scatter.set_offsets([(c + 0.5, r + 0.5) for r, c in expanded_order] or np.empty((0, 2)))
        if path:
            path_line.set_data([c + 0.5 for _, c in path], [r + 0.5 for r, _ in path])
            r, c = path[-1]
            agent_dot.set_data([c + 0.5], [r + 0.5])
            title.set_text(f"A* path | step {max(0, len(path) - 1)}/{max(0, len(path) - 1)}")
        fig.savefig("test_astar_preview.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        print("Saved static preview to test_astar_preview.png")
        return

    search_chunk = max(1, len(expanded_order) // MAX_SEARCH_FRAMES) if expanded_order else 1
    search_frames = max(1, (len(expanded_order) + search_chunk - 1) // search_chunk)
    path_frames = max(1, len(path))
    total_frames = search_frames + path_frames

    def update(frame_idx: int):
        if frame_idx < search_frames:
            expanded_count = min(len(expanded_order), (frame_idx + 1) * search_chunk)
            expanded_prefix = expanded_order[:expanded_count]
            expanded_scatter.set_offsets([(c + 0.5, r + 0.5) for r, c in expanded_prefix] or np.empty((0, 2)))
            path_line.set_data([], [])
            agent_dot.set_data([], [])
            title.set_text(f"A* search | expanded {expanded_count}/{len(expanded_order)}")
            return [expanded_scatter, path_line, agent_dot, title]

        path_idx = min(len(path), frame_idx - search_frames + 1)
        expanded_scatter.set_offsets([(c + 0.5, r + 0.5) for r, c in expanded_order] or np.empty((0, 2)))
        path_prefix = path[:path_idx]
        path_line.set_data([c + 0.5 for _, c in path_prefix], [r + 0.5 for r, _ in path_prefix])

        if path_prefix:
            r, c = path_prefix[-1]
            agent_dot.set_data([c + 0.5], [r + 0.5])
        else:
            agent_dot.set_data([], [])

        title.set_text(f"A* path | step {max(0, path_idx - 1)}/{max(0, len(path) - 1)}")
        return [expanded_scatter, path_line, agent_dot, title]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=ANIMATION_FRAME_MS,
        blit=False,
        repeat=False,
    )
    LAST_ANIMATION = ani
    fig._maze_astar_animation = ani
    plt.show()


def main() -> None:
    image_path = resolve_image_path()
    env = MazeEnvironment(image_path=image_path, maze_size=64)
    result = astar_search_debug(env.start, env.goal, lambda cell: teleport_aware_neighbors(env, cell))

    path = result["path"]
    expanded_order = result["expanded_order"]

    print(f"Image : {image_path}")
    print(f"Expanded nodes : {len(expanded_order)}")

    if not path:
        print("No path found.")
        animate_astar(env, expanded_order, path)
        return

    summarize_path(env, path)
    animate_astar(env, expanded_order, path)


if __name__ == "__main__":
    main()
