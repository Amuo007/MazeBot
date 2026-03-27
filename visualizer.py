from __future__ import annotations

from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from environment import (
    EMPTY,
    FIRE,
    START,
    GOAL,
    UNKNOWN,
    CONFUSION,
    TP_PURPLE,
    TP_RED,
    TP_GREEN,
)

Cell = Tuple[int, int]

NAME_TO_CHAR = {
    EMPTY: ".",
    FIRE: "F",
    CONFUSION: "C",
    TP_PURPLE: "P",
    TP_RED: "R",
    TP_GREEN: "G",
    START: "S",
    GOAL: "E",
    UNKNOWN: "?",
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

COL_AGENT    = np.array([1.00, 0.15, 0.15])
COL_VISITED  = np.array([0.60, 0.82, 1.00])
COL_PATH     = np.array([1.00, 0.95, 0.55])
COL_SEARCH   = np.array([0.80, 0.92, 1.00])
COL_CLOSED   = np.array([0.65, 0.82, 0.98])


def build_display(obj_matrix, env, agent) -> np.ndarray:
    n = obj_matrix.shape[0]
    disp = np.ones((n, n, 3), dtype=float)

    for r in range(n):
        for c in range(n):
            tile = obj_matrix[r, c]
            if tile != EMPTY and tile != FIRE:
                disp[r, c] = DISPLAY_COLORS.get(tile, DISPLAY_COLORS[UNKNOWN])

    # active fire
    active_fire = env.get_active_fire_cells()
    for r, c in active_fire:
        disp[r, c] = DISPLAY_COLORS[FIRE]

    # searched / expanded nodes
    if hasattr(agent, "last_search_expanded"):
        for cell in agent.last_search_expanded:
            r, c = cell
            disp[r, c] = disp[r, c] * 0.45 + COL_SEARCH * 0.55

    # closed set
    if hasattr(agent, "last_search_closed"):
        for cell in agent.last_search_closed:
            r, c = cell
            disp[r, c] = disp[r, c] * 0.55 + COL_CLOSED * 0.45

    # final chosen path
    if hasattr(agent, "current_path") and agent.current_path:
        for cell in agent.current_path[1:]:
            r, c = cell
            disp[r, c] = disp[r, c] * 0.25 + COL_PATH * 0.75

    # visited by agent
    if hasattr(agent, "memory") and hasattr(agent.memory, "visited"):
        for r, c in agent.memory.visited:
            disp[r, c] = disp[r, c] * 0.40 + COL_VISITED * 0.60

    sr, sc = env.start
    gr, gc = env.goal
    disp[sr, sc] = DISPLAY_COLORS[START]
    disp[gr, gc] = DISPLAY_COLORS[GOAL]

    ar, ac = env.position
    disp[ar, ac] = COL_AGENT

    return disp


def draw_static_walls(ax, vertical_walls, horizontal_walls, n: int) -> None:
    for r in range(n):
        for c in range(n + 1):
            if vertical_walls[r, c]:
                ax.plot([c, c], [r, r + 1], color="black", linewidth=1)

    for r in range(n + 1):
        for c in range(n):
            if horizontal_walls[r, c]:
                ax.plot([c, c + 1], [r, r], color="black", linewidth=1)


def draw_marker_labels(ax, obj_matrix) -> None:
    n = obj_matrix.shape[0]
    for r in range(n):
        for c in range(n):
            tile = obj_matrix[r, c]
            if tile != EMPTY and tile != FIRE:
                ax.text(
                    c + 0.5,
                    r + 0.56,
                    NAME_TO_CHAR.get(tile, "?"),
                    ha="center",
                    va="center",
                    fontsize=7,
                )


def show_debug_detection(img_rgb, icons) -> None:
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


def animate_episode(env, agent, max_turns: int = 10000, frame_ms: int = 120) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    n = env.maze_size

    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    im = ax.imshow(
        build_display(env.obj_matrix, env, agent),
        extent=(0, n, n, 0),
        interpolation="nearest"
    )

    draw_static_walls(ax, env.vertical_walls, env.horizontal_walls, n)
    draw_marker_labels(ax, env.obj_matrix)

    title = ax.set_title("Turn 0 | Action 0", fontsize=10)

    state = {
        "last_turn_result": None,
        "done": False,
        "pending_actions": [],
        "turn_confused": False,
        "action_in_turn": 0,
    }
    ani_holder = {"ani": None}

    def update(_frame):
        if state["done"]:
            return [im, title]

        if not state["pending_actions"]:
            state["pending_actions"] = agent.plan_turn(state["last_turn_result"])
            state["turn_confused"] = env.confused_turns_remaining > 0
            env.confused_this_turn = state["turn_confused"]
            state["action_in_turn"] = 0

        action = state["pending_actions"].pop(0)
        state["action_in_turn"] += 1

        atomic_result = env.step_one_action(action, state["turn_confused"])

        turn_finished = (
            atomic_result.is_dead
            or atomic_result.is_goal_reached
            or len(state["pending_actions"]) == 0
        )

        if turn_finished:
            turn_result = env.finish_turn(atomic_result)
            state["last_turn_result"] = turn_result
        else:
            turn_result = atomic_result

        im.set_data(build_display(env.obj_matrix, env, agent))
        phase = (env.total_actions_executed // 5) % len(env.fire_phase_sets)

        shown_turn = env.turns_taken if turn_finished else env.turns_taken + 1

        if turn_result.is_goal_reached:
            title.set_text(
                f"GOAL | Turn {env.turns_taken} | Action {state['action_in_turn']} | "
                f"Steps={env.total_actions_executed} | Pos={turn_result.current_position}"
            )
            title.set_color("green")
            state["done"] = True
            if ani_holder["ani"] is not None:
                ani_holder["ani"].event_source.stop()

        elif turn_result.is_dead:
            title.set_text(
                f"DEAD | Turn {env.turns_taken} | Action {state['action_in_turn']} | "
                f"Steps={env.total_actions_executed} | Respawn={turn_result.current_position}"
            )
            title.set_color("red")

        else:
            title.set_text(
                f"Turn {shown_turn} | Action {state['action_in_turn']} | "
                f"Steps={env.total_actions_executed} | "
                f"Pos={turn_result.current_position} | "
                f"Walls={turn_result.wall_hits} | "
                f"Confused={turn_result.is_confused} | "
                f"Teleported={turn_result.teleported} | "
                f"FirePhase={phase}"
            )
            title.set_color("black")

        return [im, title]

    ani_holder["ani"] = animation.FuncAnimation(
        fig,
        update,
        frames=max_turns * 5,
        interval=frame_ms,
        blit=False,
        repeat=False,
    )

    plt.tight_layout()
    plt.show()