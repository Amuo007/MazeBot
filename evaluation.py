from __future__ import annotations

import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from environment import ACTIONS_PER_TURN, Action, MazeEnvironment, TurnResult
from exploration import BlindKnowledge, Cell
from qlearning import QLearner
from route_execution import RouteExecutionAgent, build_endgame_agent

DEFAULT_PHASE2_REPORT_PATH = "phase2_report.json"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _empty_action_counts() -> Dict[str, int]:
    return {action.name: 0 for action in Action}


def _counter_to_action_counts(counter: Counter) -> Dict[str, int]:
    counts = _empty_action_counts()
    for action_name, count in counter.items():
        counts[action_name] = int(count)
    return counts


def _execute_turn_with_trace(
    env: MazeEnvironment,
    actions: List[Action],
) -> tuple[TurnResult, List[Dict[str, Any]]]:
    if not (1 <= len(actions) <= ACTIONS_PER_TURN):
        raise ValueError(f"actions must contain between 1 and {ACTIONS_PER_TURN} actions")

    turn_confused = env.confused_turns_remaining > 0
    env.confused_this_turn = turn_confused
    final_result = TurnResult(current_position=env.position)
    action_trace: List[Dict[str, Any]] = []

    for action_index, planned_action in enumerate(actions, start=1):
        action_started_confused = turn_confused or env.confused_this_turn
        effective_action = env.apply_confusion(planned_action) if action_started_confused else planned_action
        position_before = env.position

        atomic_result = env.step_one_action(planned_action, turn_confused)
        action_trace.append(
            {
                "action_in_turn": action_index,
                "position_before": position_before,
                "position_after": atomic_result.current_position,
                "planned_action": planned_action.name,
                "effective_action": effective_action.name,
                "wall_hit": bool(atomic_result.wall_hits),
                "teleported": bool(atomic_result.teleported),
                "is_dead": bool(atomic_result.is_dead),
                "is_goal_reached": bool(atomic_result.is_goal_reached),
                "is_confused": bool(atomic_result.is_confused),
            }
        )

        final_result.wall_hits += atomic_result.wall_hits
        final_result.current_position = atomic_result.current_position
        final_result.is_dead = atomic_result.is_dead
        final_result.is_confused = final_result.is_confused or atomic_result.is_confused
        final_result.is_goal_reached = atomic_result.is_goal_reached
        final_result.teleported = final_result.teleported or atomic_result.teleported
        final_result.actions_executed += atomic_result.actions_executed

        if atomic_result.is_dead or atomic_result.is_goal_reached:
            break

    return env.finish_turn(final_result), action_trace


def _run_phase2_episode(
    env: MazeEnvironment,
    agent: RouteExecutionAgent,
    max_turns: int,
) -> Dict[str, Any]:
    env.reset()
    agent.reset_episode()

    last_result: Optional[TurnResult] = None
    total_wall_hits = 0
    teleports_triggered = 0
    confused_turns = 0
    actions_planned = 0
    executed_steps = 0
    movement_steps_executed = 0
    wait_actions_executed = 0
    wait_actions_planned = 0
    teleport_steps = 0
    death_steps = 0
    goal_steps = 0
    turns_with_wait_actions = 0
    turns_with_teleport = 0
    turns_with_wall_hit = 0

    planned_action_counts: Counter = Counter()
    executed_action_counts: Counter = Counter()
    effective_action_counts: Counter = Counter()
    meta_action_counts: Counter = Counter()

    for _turn in range(max_turns):
        actions = agent.plan_turn(last_result)
        actions_planned += len(actions)
        chosen_meta_action = getattr(agent, "_last_meta_action", None)
        if chosen_meta_action is not None:
            meta_action_counts[chosen_meta_action.name] += 1

        planned_action_counts.update(action.name for action in actions)
        wait_actions_planned += sum(action == Action.WAIT for action in actions)
        turns_with_wait_actions += int(any(action == Action.WAIT for action in actions))

        last_result, action_trace = _execute_turn_with_trace(env, actions)
        total_wall_hits += last_result.wall_hits
        teleports_triggered += int(last_result.teleported)
        confused_turns += int(last_result.is_confused)
        turns_with_teleport += int(last_result.teleported)
        turns_with_wall_hit += int(last_result.wall_hits > 0)

        for action_info in action_trace:
            executed_steps += 1
            executed_action_counts[action_info["planned_action"]] += 1
            effective_action_counts[action_info["effective_action"]] += 1
            wait_actions_executed += int(action_info["planned_action"] == Action.WAIT.name)
            movement_steps_executed += int(action_info["planned_action"] != Action.WAIT.name)
            teleport_steps += int(action_info["teleported"])
            death_steps += int(action_info["is_dead"])
            goal_steps += int(action_info["is_goal_reached"])

        if last_result.is_goal_reached:
            break

    return {
        "actions_planned": actions_planned,
        "executed_steps": executed_steps,
        "movement_steps_executed": movement_steps_executed,
        "total_wall_hits": total_wall_hits,
        "teleports_triggered": teleports_triggered,
        "confused_turns": confused_turns,
        "wait_actions_planned": wait_actions_planned,
        "wait_actions_executed": wait_actions_executed,
        "teleport_steps": teleport_steps,
        "death_steps": death_steps,
        "goal_steps": goal_steps,
        "turns_with_wait_actions": turns_with_wait_actions,
        "turns_with_teleport": turns_with_teleport,
        "turns_with_wall_hit": turns_with_wall_hit,
        "planned_action_counts": _counter_to_action_counts(planned_action_counts),
        "executed_action_counts": _counter_to_action_counts(executed_action_counts),
        "effective_action_counts": _counter_to_action_counts(effective_action_counts),
        "meta_action_counts": {name: int(count) for name, count in meta_action_counts.items()},
    }


def _build_phase2_report(
    env: MazeEnvironment,
    agent: RouteExecutionAgent,
    knowledge: BlindKnowledge,
    discovered_path: List[Cell],
    run_summary: Dict[str, Any],
    max_turns: int,
    phase1_exploration_episodes: Optional[int],
    qtable_state_count: Optional[int],
) -> Dict[str, Any]:
    stats = env.get_episode_stats()
    total_cells_visited = len(env.cells_visited)
    unique_cells_visited = len(env.unique_cells)
    turns_taken = stats["turns_taken"]
    total_actions_executed = stats["total_actions_executed"]
    total_steps_including_all_events = run_summary["executed_steps"]
    movement_steps_executed = run_summary["movement_steps_executed"]
    path_length_excluding_teleports = max(0, total_cells_visited - 1)
    route_length_cells = max(0, len(discovered_path) - 1)
    route_progress_index = min(agent.route_progress_index, route_length_cells)

    success = bool(stats["goal_reached"])
    death_rate = (stats["deaths"] / turns_taken) if turns_taken else 0.0
    exploration_efficiency = (unique_cells_visited / total_cells_visited) if total_cells_visited else 0.0
    route_completion_ratio = (route_progress_index / route_length_cells) if route_length_cells else float(success)

    return {
        "report_type": "phase_2_endgame_evaluation",
        "phase_scope": "phase_2_only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "maze_image": env.image_path,
        "success": success,
        "terminated_reason": "goal_reached" if success else "turn_limit_reached",
        "turn_limit": max_turns,
        "phase1_context": {
            "exploration_episodes": phase1_exploration_episodes,
            "discovered_cells": len(knowledge.visited),
            "discovered_walls": len(knowledge.blocked_edges) // 2,
            "discovered_teleports": len(knowledge.teleport_pairs) // 2,
            "discovered_route_length_cells": route_length_cells,
        },
        "policy_context": {
            "qtable_states_loaded": qtable_state_count,
            "phase2_epsilon": 0.0,
            "rl_training_performed_in_run_py": False,
        },
        "positions": {
            "start": list(env.start),
            "goal": list(env.goal),
            "final": list(env.position),
        },
        "raw_counts": {
            "turns_taken": turns_taken,
            "total_actions_executed": total_actions_executed,
            "total_steps_including_repeats_teleports_and_death_steps": total_steps_including_all_events,
            "movement_steps_executed_excluding_waits": movement_steps_executed,
            "actions_planned": run_summary["actions_planned"],
            "deaths": stats["deaths"],
            "death_steps": run_summary["death_steps"],
            "wall_hits": run_summary["total_wall_hits"],
            "teleports_triggered": run_summary["teleports_triggered"],
            "teleport_steps": run_summary["teleport_steps"],
            "confused_turns": run_summary["confused_turns"],
            "wait_actions_planned": run_summary["wait_actions_planned"],
            "wait_actions_executed": run_summary["wait_actions_executed"],
            "goal_steps": run_summary["goal_steps"],
            "turns_with_wait_actions": run_summary["turns_with_wait_actions"],
            "turns_with_teleport": run_summary["turns_with_teleport"],
            "turns_with_wall_hit": run_summary["turns_with_wall_hit"],
            "total_cells_visited_including_duplicates": total_cells_visited,
            "unique_cells_visited": unique_cells_visited,
            "path_length_excluding_teleports": path_length_excluding_teleports,
        },
        "action_breakdown": {
            "planned_action_counts": run_summary["planned_action_counts"],
            "executed_action_counts": run_summary["executed_action_counts"],
            "effective_action_counts_after_confusion": run_summary["effective_action_counts"],
            "unique_planned_action_types_used": [
                action_name
                for action_name, count in run_summary["planned_action_counts"].items()
                if count > 0
            ],
            "unique_executed_action_types_used": [
                action_name
                for action_name, count in run_summary["executed_action_counts"].items()
                if count > 0
            ],
        },
        "policy_breakdown": {
            "meta_action_counts": run_summary["meta_action_counts"],
        },
        "route_summary": {
            "discovered_route_length_cells": route_length_cells,
            "route_progress_index": route_progress_index,
            "route_completion_ratio": route_completion_ratio,
            "discovered_cells_from_phase_1": len(knowledge.visited),
            "discovered_walls_from_phase_1": len(knowledge.blocked_edges) // 2,
            "discovered_teleports_from_phase_1": len(knowledge.teleport_pairs) // 2,
        },
        "derived_metrics": {
            "success_rate": 1.0 if success else 0.0,
            "avg_turns": float(turns_taken) if success else None,
            "avg_deaths": float(stats["deaths"]),
            "avg_path_length": float(path_length_excluding_teleports) if success else None,
            "death_rate": death_rate,
            "exploration_efficiency": exploration_efficiency,
            "avg_actions_per_turn": (total_actions_executed / turns_taken) if turns_taken else 0.0,
        },
        "notes": [
            "This report is generated from the phase 2 endgame execution only.",
            "total_actions_executed and total_steps_including_repeats_teleports_and_death_steps count every executed primitive action, including WAIT.",
            "movement_steps_executed_excluding_waits counts only executed movement steps and excludes WAIT actions.",
            "Teleport jumps are excluded from path_length_excluding_teleports.",
            "Five-episode instructor averages are not computed here; this file records the single phase 2 run.",
        ],
    }


def write_phase2_report(
    env: MazeEnvironment,
    agent: RouteExecutionAgent,
    knowledge: BlindKnowledge,
    discovered_path: List[Cell],
    run_summary: Dict[str, Any],
    report_path: str = DEFAULT_PHASE2_REPORT_PATH,
    max_turns: int = 10000,
    phase1_exploration_episodes: Optional[int] = None,
    qtable_state_count: Optional[int] = None,
) -> Dict[str, Any]:
    report = _json_ready(_build_phase2_report(
        env,
        agent,
        knowledge,
        discovered_path,
        run_summary,
        max_turns=max_turns,
        phase1_exploration_episodes=phase1_exploration_episodes,
        qtable_state_count=qtable_state_count,
    ))

    report_file = Path(report_path)
    with report_file.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")

    return report


def evaluate_phase2_and_write_report(
    image_path: str,
    maze_size: int,
    knowledge: BlindKnowledge,
    discovered_path: List[Cell],
    qlearner: QLearner,
    report_path: str = DEFAULT_PHASE2_REPORT_PATH,
    max_turns: int = 10000,
    phase1_exploration_episodes: Optional[int] = None,
) -> Dict[str, Any]:
    eval_env = MazeEnvironment(image_path=image_path, maze_size=maze_size)
    eval_qlearner = copy.deepcopy(qlearner)
    eval_qlearner.epsilon = 0.0
    eval_agent = build_endgame_agent(eval_env, knowledge, discovered_path, eval_qlearner)

    run_summary = _run_phase2_episode(eval_env, eval_agent, max_turns=max_turns)
    return write_phase2_report(
        eval_env,
        eval_agent,
        knowledge,
        discovered_path,
        run_summary,
        report_path=report_path,
        max_turns=max_turns,
        phase1_exploration_episodes=phase1_exploration_episodes,
        qtable_state_count=len(qlearner.q_table),
    )
