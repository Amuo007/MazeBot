from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from environment import MazeEnvironment, TurnResult
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

    for _turn in range(max_turns):
        actions = agent.plan_turn(last_result)
        actions_planned += len(actions)

        last_result = env.step(actions)
        total_wall_hits += last_result.wall_hits
        teleports_triggered += int(last_result.teleported)
        confused_turns += int(last_result.is_confused)

        if last_result.is_goal_reached:
            break

    return {
        "actions_planned": actions_planned,
        "total_wall_hits": total_wall_hits,
        "teleports_triggered": teleports_triggered,
        "confused_turns": confused_turns,
    }


def _build_phase2_report(
    env: MazeEnvironment,
    agent: RouteExecutionAgent,
    knowledge: BlindKnowledge,
    discovered_path: List[Cell],
    run_summary: Dict[str, Any],
    max_turns: int,
) -> Dict[str, Any]:
    stats = env.get_episode_stats()
    total_cells_visited = len(env.cells_visited)
    unique_cells_visited = len(env.unique_cells)
    turns_taken = stats["turns_taken"]
    total_actions_executed = stats["total_actions_executed"]
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
        "positions": {
            "start": list(env.start),
            "goal": list(env.goal),
            "final": list(env.position),
        },
        "raw_counts": {
            "turns_taken": turns_taken,
            "total_actions_executed": total_actions_executed,
            "actions_planned": run_summary["actions_planned"],
            "deaths": stats["deaths"],
            "wall_hits": run_summary["total_wall_hits"],
            "teleports_triggered": run_summary["teleports_triggered"],
            "confused_turns": run_summary["confused_turns"],
            "total_cells_visited_including_duplicates": total_cells_visited,
            "unique_cells_visited": unique_cells_visited,
            "path_length_excluding_teleports": path_length_excluding_teleports,
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
        },
        "notes": [
            "This report is generated from the phase 2 endgame execution only.",
            "Teleport jumps are excluded from path_length_excluding_teleports.",
            "Five-episode instructor averages are not computed here; this file records the single phase 2 run.",
        ],
    }


def evaluate_phase2_and_write_report(
    image_path: str,
    maze_size: int,
    knowledge: BlindKnowledge,
    discovered_path: List[Cell],
    qlearner: QLearner,
    report_path: str = DEFAULT_PHASE2_REPORT_PATH,
    max_turns: int = 10000,
) -> Dict[str, Any]:
    eval_env = MazeEnvironment(image_path=image_path, maze_size=maze_size)
    eval_qlearner = copy.deepcopy(qlearner)
    eval_qlearner.epsilon = 0.0
    eval_agent = build_endgame_agent(eval_env, knowledge, discovered_path, eval_qlearner)

    run_summary = _run_phase2_episode(eval_env, eval_agent, max_turns=max_turns)
    report = _json_ready(_build_phase2_report(
        eval_env,
        eval_agent,
        knowledge,
        discovered_path,
        run_summary,
        max_turns=max_turns,
    ))

    report_file = Path(report_path)
    with report_file.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")

    return report
