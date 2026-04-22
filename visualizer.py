from __future__ import annotations

from collections import Counter

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from environment import (
    ACTIONS_PER_TURN,
    Action,
    EMPTY,
    FIRE,
    FIRE_CENTER,
    START,
    GOAL,
    ONE_WAY_GATE,
    UNKNOWN,
    CONFUSION,
    TP_PURPLE,
    TP_RED,
    TP_GREEN,
    TP_LAVENDER,
    TurnResult,
)

NAME_TO_CHAR = {
    EMPTY: ".",
    FIRE: "F",
    FIRE_CENTER: "O",
    CONFUSION: "C",
    TP_PURPLE: "P",
    TP_RED: "R",
    TP_GREEN: "G",
    TP_LAVENDER: "L",
    ONE_WAY_GATE: "G",
    START: "S",
    GOAL: "E",
    UNKNOWN: "?",
}

DISPLAY_COLORS = {
    EMPTY:      np.array([1.00, 1.00, 1.00]),
    FIRE:       np.array([255, 145, 76]) / 255.0,
    FIRE_CENTER:np.array([253, 183, 140]) / 255.0,
    CONFUSION:  np.array([255, 222, 89]) / 255.0,
    TP_PURPLE:  np.array([140, 82, 255]) / 255.0,
    TP_RED:     np.array([255, 49, 50]) / 255.0,
    TP_GREEN:   np.array([1, 191, 99]) / 255.0,
    TP_LAVENDER:np.array([226, 169, 241]) / 255.0,
    ONE_WAY_GATE:np.array([126, 217, 255]) / 255.0,
    START:      np.array([15, 192, 223]) / 255.0,
    GOAL:       np.array([0, 74, 173]) / 255.0,
    UNKNOWN:    np.array([0.75, 0.75, 0.75]),
}

COL_AGENT    = np.array([1.00, 0.15, 0.15])
COL_VISITED  = np.array([0.60, 0.82, 1.00])
COL_PATH     = np.array([1.00, 0.95, 0.55])

ACTION_LABELS = {
    "MOVE_UP": "U",
    "MOVE_DOWN": "D",
    "MOVE_LEFT": "L",
    "MOVE_RIGHT": "R",
    "WAIT": "W",
}


def format_action(action) -> str:
    return ACTION_LABELS.get(getattr(action, "name", ""), str(action))


def get_display_obj_matrix(env, agent):
    return getattr(agent, "display_obj_matrix", getattr(agent, "obj_matrix", env.obj_matrix))


def get_display_walls(env, agent):
    vertical_walls = getattr(agent, "display_vertical_walls", getattr(agent, "vertical_walls", env.vertical_walls))
    horizontal_walls = getattr(agent, "display_horizontal_walls", getattr(agent, "horizontal_walls", env.horizontal_walls))
    return vertical_walls, horizontal_walls


def build_display(obj_matrix, env, agent) -> np.ndarray:
    n = obj_matrix.shape[0]
    disp = np.ones((n, n, 3), dtype=float)

    for r in range(n):
        for c in range(n):
            tile = obj_matrix[r, c]
            if tile not in (EMPTY, FIRE):
                disp[r, c] = DISPLAY_COLORS.get(tile, DISPLAY_COLORS[UNKNOWN])

    active_fire = env.get_active_fire_cells()
    for r, c in active_fire:
        disp[r, c] = DISPLAY_COLORS[FIRE]

    if hasattr(agent, "current_path") and agent.current_path:
        for cell in agent.current_path[1:]:
            r, c = cell
            disp[r, c] = disp[r, c] * 0.25 + COL_PATH * 0.75

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
            if tile not in (EMPTY, FIRE, UNKNOWN):
                ax.text(
                    c + 0.5,
                    r + 0.56,
                    NAME_TO_CHAR.get(tile, "?"),
                    ha="center",
                    va="center",
                    fontsize=7,
                )


def _empty_action_counts() -> dict[str, int]:
    return {action.name: 0 for action in Action}


def _counter_to_action_counts(counter: Counter) -> dict[str, int]:
    counts = _empty_action_counts()
    for action_name, count in counter.items():
        counts[action_name] = int(count)
    return counts


def animate_episode(env, agent, max_turns: int = 10000, frame_ms: int = 120) -> dict:
    fig, ax = plt.subplots(figsize=(10, 10))
    n = env.maze_size
    fig.subplots_adjust(right=0.78)
    display_obj_matrix = get_display_obj_matrix(env, agent)
    display_vertical_walls, display_horizontal_walls = get_display_walls(env, agent)

    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    im = ax.imshow(
        build_display(display_obj_matrix, env, agent),
        extent=(0, n, n, 0),
        interpolation="nearest"
    )

    draw_static_walls(ax, display_vertical_walls, display_horizontal_walls, n)
    draw_marker_labels(ax, display_obj_matrix)

    title = ax.set_title("Turn 0 | Action 0", fontsize=10)
    action_box = ax.text(
        1.02,
        0.98,
        "Turn actions\n-",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", alpha=0.95),
    )

    state = {
        "last_turn_result": None,
        "done": False,
        "pending_actions": [],
        "current_turn_actions": [],
        "turn_confused": False,
        "action_in_turn": 0,
        "aggregated_turn_result": None,
    }
    ani_holder = {"ani": None}
    run_summary = {
        "actions_planned": 0,
        "executed_steps": 0,
        "movement_steps_executed": 0,
        "total_wall_hits": 0,
        "teleports_triggered": 0,
        "confused_turns": 0,
        "wait_actions_planned": 0,
        "wait_actions_executed": 0,
        "teleport_steps": 0,
        "death_steps": 0,
        "goal_steps": 0,
        "turns_with_wait_actions": 0,
        "turns_with_teleport": 0,
        "turns_with_wall_hit": 0,
        "planned_action_counts": Counter(),
        "executed_action_counts": Counter(),
        "effective_action_counts": Counter(),
        "meta_action_counts": Counter(),
    }

    def update_action_box() -> None:
        actions = state["current_turn_actions"]
        if not actions:
            action_box.set_text("Turn actions\n-")
            return

        lines = ["Turn actions"]
        current_index = max(0, state["action_in_turn"] - 1)
        for index, action in enumerate(actions):
            marker = ">" if index == current_index else " "
            lines.append(f"{marker} {index + 1}: {format_action(action)}")
        action_box.set_text("\n".join(lines))

    def update(_frame):
        if state["done"]:
            update_action_box()
            return [im, title, action_box]

        if not state["pending_actions"]:
            state["pending_actions"] = agent.plan_turn(state["last_turn_result"])
            state["current_turn_actions"] = list(state["pending_actions"])
            run_summary["actions_planned"] += len(state["current_turn_actions"])
            run_summary["planned_action_counts"].update(action.name for action in state["current_turn_actions"])
            run_summary["wait_actions_planned"] += sum(action == Action.WAIT for action in state["current_turn_actions"])
            run_summary["turns_with_wait_actions"] += int(any(action == Action.WAIT for action in state["current_turn_actions"]))
            chosen_meta_action = getattr(agent, "_last_meta_action", None)
            if chosen_meta_action is not None:
                run_summary["meta_action_counts"][chosen_meta_action.name] += 1
            state["turn_confused"] = env.confused_turns_remaining > 0
            env.confused_this_turn = state["turn_confused"]
            state["action_in_turn"] = 0
            state["aggregated_turn_result"] = TurnResult(current_position=env.position)
            update_action_box()

        action = state["pending_actions"].pop(0)
        state["action_in_turn"] += 1
        update_action_box()

        action_started_confused = state["turn_confused"] or env.confused_this_turn
        effective_action = env.apply_confusion(action) if action_started_confused else action
        atomic_result = env.step_one_action(action, state["turn_confused"])
        run_summary["executed_steps"] += 1
        run_summary["executed_action_counts"][action.name] += 1
        run_summary["effective_action_counts"][effective_action.name] += 1
        run_summary["wait_actions_executed"] += int(action == Action.WAIT)
        run_summary["movement_steps_executed"] += int(action != Action.WAIT)
        run_summary["teleport_steps"] += int(atomic_result.teleported)
        run_summary["death_steps"] += int(atomic_result.is_dead)
        run_summary["goal_steps"] += int(atomic_result.is_goal_reached)

        aggregated_turn_result = state["aggregated_turn_result"]
        aggregated_turn_result.wall_hits += atomic_result.wall_hits
        aggregated_turn_result.current_position = atomic_result.current_position
        aggregated_turn_result.is_dead = atomic_result.is_dead
        aggregated_turn_result.is_confused = aggregated_turn_result.is_confused or atomic_result.is_confused
        aggregated_turn_result.is_goal_reached = atomic_result.is_goal_reached
        aggregated_turn_result.teleported = aggregated_turn_result.teleported or atomic_result.teleported
        aggregated_turn_result.actions_executed += atomic_result.actions_executed

        turn_finished = (
            atomic_result.is_dead
            or atomic_result.is_goal_reached
            or len(state["pending_actions"]) == 0
        )

        if turn_finished:
            turn_result = env.finish_turn(aggregated_turn_result)
            run_summary["total_wall_hits"] += turn_result.wall_hits
            run_summary["teleports_triggered"] += int(turn_result.teleported)
            run_summary["confused_turns"] += int(turn_result.is_confused)
            run_summary["turns_with_teleport"] += int(turn_result.teleported)
            run_summary["turns_with_wall_hit"] += int(turn_result.wall_hits > 0)
            state["last_turn_result"] = turn_result
            state["aggregated_turn_result"] = None
            state["pending_actions"] = []
        else:
            turn_result = aggregated_turn_result

        im.set_data(build_display(get_display_obj_matrix(env, agent), env, agent))
        phase = (env.total_actions_executed // ACTIONS_PER_TURN) % len(env.fire_phase_sets)

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

        return [im, title, action_box]

    ani_holder["ani"] = animation.FuncAnimation(
        fig,
        update,
        frames=max_turns * ACTIONS_PER_TURN,
        interval=frame_ms,
        blit=False,
        repeat=False,
    )

    plt.tight_layout()
    plt.show()

    return {
        "actions_planned": run_summary["actions_planned"],
        "executed_steps": run_summary["executed_steps"],
        "movement_steps_executed": run_summary["movement_steps_executed"],
        "total_wall_hits": run_summary["total_wall_hits"],
        "teleports_triggered": run_summary["teleports_triggered"],
        "confused_turns": run_summary["confused_turns"],
        "wait_actions_planned": run_summary["wait_actions_planned"],
        "wait_actions_executed": run_summary["wait_actions_executed"],
        "teleport_steps": run_summary["teleport_steps"],
        "death_steps": run_summary["death_steps"],
        "goal_steps": run_summary["goal_steps"],
        "turns_with_wait_actions": run_summary["turns_with_wait_actions"],
        "turns_with_teleport": run_summary["turns_with_teleport"],
        "turns_with_wall_hit": run_summary["turns_with_wall_hit"],
        "planned_action_counts": _counter_to_action_counts(run_summary["planned_action_counts"]),
        "executed_action_counts": _counter_to_action_counts(run_summary["executed_action_counts"]),
        "effective_action_counts": _counter_to_action_counts(run_summary["effective_action_counts"]),
        "meta_action_counts": {name: int(count) for name, count in run_summary["meta_action_counts"].items()},
    }
