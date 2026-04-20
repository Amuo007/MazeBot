from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from environment import Action, CONFUSION, MazeEnvironment

from .config import (
    MAZE_SIZE,
    MAX_PHYSICAL_STEPS,
    PRINT_EVERY,
    STALL_FRONTIER_TRIGGER,
    SUCCESS_STREAK_TO_STOP,
    TRAIN_EPISODES,
    TRAIN_PRINT_EVERY,
    TRAINING_STEP_BUDGET,
)
from .knowledge import (
    BlindKnowledge,
    Cell,
    adjacent,
    delta_to_action,
    fire_time_mod,
    invert_action,
)
from .planning import (
    PlannedStep,
    pick_frontier_probe,
    plan_timed_route,
)


@dataclass
class StepOutcome:
    event: str
    moved: bool
    is_dead: bool
    is_goal: bool
    wall_hit: bool
    discovered_new_cell: bool
    discovered_new_info: bool
    teleported: bool


class BlindExplorer:
    def __init__(
        self,
        env: MazeEnvironment,
        knowledge: Optional[BlindKnowledge] = None,
    ):
        self.env = env
        self.knowledge = knowledge if knowledge is not None else BlindKnowledge(env.maze_size, env.start, env.goal)
        self.knowledge.start_new_episode()

        self.current: Cell = env.start
        self.physical_steps = 0
        self.deaths = 0
        self.wall_bumps = 0
        self.teleports_taken = 0
        self.last_new_info_step = 0

    def _observe_cell(self, cell: Cell) -> bool:
        was_new = cell not in self.knowledge.visited
        self.knowledge.visited.add(cell)
        self.knowledge.tile_of[cell] = int(self.env.obj_matrix[cell])
        return was_new

    def _observe_confusion_if_entered(self, cell: Cell) -> None:
        if int(self.env.obj_matrix[cell]) == CONFUSION:
            self.knowledge.confusion_cells.add(cell)

    def _mark_new_info(self) -> None:
        self.last_new_info_step = self.physical_steps

    def _log_progress(self, event: str, target: Optional[Cell]) -> None:
        if self.physical_steps % PRINT_EVERY == 0:
            print(
                f"[EXPLORE] steps={self.physical_steps:5d} "
                f"visited={len(self.knowledge.visited):4d} "
                f"walls={len(self.knowledge.blocked_edges) // 2:4d} "
                f"deaths={self.deaths:3d} "
                f"tp={self.teleports_taken:3d} "
                f"target={target} event={event}"
            )

    def _step_physical(self, next_cell: Cell) -> StepOutcome:
        if not adjacent(self.current, next_cell):
            return StepOutcome(
                event="diverged",
                moved=False,
                is_dead=False,
                is_goal=False,
                wall_hit=False,
                discovered_new_cell=False,
                discovered_new_info=False,
                teleported=False,
            )

        planned_action = delta_to_action(self.current, next_cell)
        action_to_send = invert_action(planned_action) if self.env.confused_turns_remaining > 0 else planned_action

        before = self.current
        result = self.env.step([action_to_send])
        self.physical_steps += 1

        if result.is_dead:
            death_cell = self.knowledge.teleport_pairs.get(next_cell, next_cell)
            death_phase = self.env.get_fire_phase()
            self.knowledge.dangerous.add(death_cell)
            discovered_new_info = self.knowledge.mark_deadly_phase(death_cell, death_phase)
            if discovered_new_info:
                self._mark_new_info()
            self.current = self.env.start
            self.deaths += 1
            return StepOutcome(
                event="dead",
                moved=False,
                is_dead=True,
                is_goal=False,
                wall_hit=False,
                discovered_new_cell=False,
                discovered_new_info=True,
                teleported=result.teleported,
            )

        if result.wall_hits > 0:
            discovered_new_info = self.knowledge.mark_blocked(before, next_cell)
            if discovered_new_info:
                self._mark_new_info()
            self.wall_bumps += 1
            return StepOutcome(
                event="wall",
                moved=False,
                is_dead=False,
                is_goal=False,
                wall_hit=True,
                discovered_new_cell=False,
                discovered_new_info=discovered_new_info,
                teleported=False,
            )

        landing = result.current_position
        discovered_new_info = self.knowledge.mark_open(before, next_cell)
        discovered_new_cell = self._observe_cell(next_cell)
        self._observe_confusion_if_entered(next_cell)

        if result.teleported:
            self.knowledge.teleport_pairs[next_cell] = landing
            self.knowledge.teleport_pairs[landing] = next_cell
            discovered_new_cell = self._observe_cell(landing) or discovered_new_cell
            discovered_new_info = True
            self.teleports_taken += 1

        if discovered_new_cell or discovered_new_info:
            self._mark_new_info()

        self.current = landing
        return StepOutcome(
            event="moved",
            moved=True,
            is_dead=False,
            is_goal=(self.current == self.knowledge.goal or result.is_goal_reached),
            wall_hit=False,
            discovered_new_cell=discovered_new_cell,
            discovered_new_info=discovered_new_info,
            teleported=result.teleported,
        )

    def _wait_physical(self) -> StepOutcome:
        before = self.current
        result = self.env.step([Action.WAIT])
        self.physical_steps += 1

        if result.is_dead:
            death_phase = self.env.get_fire_phase()
            self.knowledge.dangerous.add(before)
            discovered_new_info = self.knowledge.mark_deadly_phase(before, death_phase)
            if discovered_new_info:
                self._mark_new_info()
            self.current = self.env.start
            self.deaths += 1
            return StepOutcome(
                event="dead",
                moved=False,
                is_dead=True,
                is_goal=False,
                wall_hit=False,
                discovered_new_cell=False,
                discovered_new_info=True,
                teleported=result.teleported,
            )

        discovered_new_cell = False
        discovered_new_info = False
        landing = result.current_position

        if result.teleported and landing != before:
            self.knowledge.teleport_pairs[before] = landing
            self.knowledge.teleport_pairs[landing] = before
            discovered_new_cell = self._observe_cell(landing)
            discovered_new_info = True
            self._mark_new_info()
            self.teleports_taken += 1
            self.current = landing
            return StepOutcome(
                event="diverged",
                moved=True,
                is_dead=False,
                is_goal=(self.current == self.knowledge.goal or result.is_goal_reached),
                wall_hit=False,
                discovered_new_cell=discovered_new_cell,
                discovered_new_info=discovered_new_info,
                teleported=True,
            )

        self.current = before
        return StepOutcome(
            event="wait",
            moved=False,
            is_dead=False,
            is_goal=(self.current == self.knowledge.goal or result.is_goal_reached),
            wall_hit=False,
            discovered_new_cell=False,
            discovered_new_info=False,
            teleported=False,
        )

    def _follow_timed_plan(
        self,
        plan: List[PlannedStep],
        target: Optional[Cell],
        max_steps: int,
    ) -> str:
        for planned_step in plan:
            if self.physical_steps >= max_steps:
                return "budget"

            if planned_step.step_cell is None:
                outcome = self._wait_physical()
            else:
                outcome = self._step_physical(planned_step.step_cell)

            self._log_progress(outcome.event, target)
            if outcome.is_goal or self.current == self.knowledge.goal:
                return "goal"
            if outcome.event not in {"moved", "wait"}:
                return "blocked"

        return "arrived"

    def run(self, max_steps: int = MAX_PHYSICAL_STEPS) -> bool:
        while self.physical_steps < max_steps:
            if self.current == self.knowledge.goal:
                return True

            stalled = (self.physical_steps - self.last_new_info_step) >= STALL_FRONTIER_TRIGGER
            if stalled:
                probe = pick_frontier_probe(self.knowledge, self.current)
                if probe is not None:
                    probe_from, probe_to = probe
                    plan = plan_timed_route(
                        self.knowledge,
                        self.current,
                        fire_time_mod(self.env),
                        probe_from,
                        optimistic=False,
                    )
                    if plan is not None:
                        route_status = self._follow_timed_plan(plan, target=probe_to, max_steps=max_steps)
                        if route_status == "budget":
                            return False
                        if route_status == "goal":
                            return True

                        if self.current == probe_from:
                            probe_plan = plan_timed_route(
                                self.knowledge,
                                self.current,
                                fire_time_mod(self.env),
                                probe_to,
                                optimistic=True,
                            )
                            if probe_plan is not None:
                                probe_status = self._follow_timed_plan(
                                    probe_plan,
                                    target=probe_to,
                                    max_steps=max_steps,
                                )
                                if probe_status == "budget":
                                    return False
                                if probe_status == "goal":
                                    return True
                                continue

            plan = plan_timed_route(
                self.knowledge,
                self.current,
                fire_time_mod(self.env),
                self.knowledge.goal,
                optimistic=True,
            )
            if plan is None:
                return False
            route_status = self._follow_timed_plan(plan, target=self.knowledge.goal, max_steps=max_steps)
            if route_status == "budget":
                return False
            if route_status == "goal":
                return True

        return self.current == self.knowledge.goal


def build_blind_knowledge(image_path: str) -> Tuple[BlindKnowledge, int]:
    env = MazeEnvironment(image_path=image_path, maze_size=MAZE_SIZE)
    learned_knowledge = BlindKnowledge(env.maze_size, env.start, env.goal)
    best_knowledge: Optional[BlindKnowledge] = None
    best_reached = False
    best_score = -1

    print("Building blind exploration memory...")
    print("  (episodes share discovered map/fire memory)\n")

    success_streak = 0
    episodes_ran = 0

    for episode in range(1, TRAIN_EPISODES + 1):
        env.reset()
        explorer = BlindExplorer(env, knowledge=learned_knowledge)
        reached = explorer.run(max_steps=TRAINING_STEP_BUDGET)
        episodes_ran = episode

        if reached:
            success_streak += 1
        else:
            success_streak = 0

        episode_score = len(learned_knowledge.visited) + len(learned_knowledge.blocked_edges) // 2
        if (
            best_knowledge is None
            or (reached and not best_reached)
            or (reached == best_reached and episode_score > best_score)
        ):
            best_knowledge = learned_knowledge.clone()
            best_reached = reached
            best_score = episode_score

        if episode == 1 or episode % TRAIN_PRINT_EVERY == 0 or reached:
            print(
                f"[EXPLORE] ep={episode:03d} reached={reached!s:5s} "
                f"steps={explorer.physical_steps:4d} "
                f"mapped={len(learned_knowledge.visited):4d} "
                f"walls={len(learned_knowledge.blocked_edges) // 2:4d} "
                f"deaths={explorer.deaths:3d}"
            )

        if success_streak >= SUCCESS_STREAK_TO_STOP:
            print(f"[EXPLORE] early stop after {success_streak} consecutive successes")
            break

    print()
    return best_knowledge if best_knowledge is not None else learned_knowledge.clone(), episodes_ran
