from __future__ import annotations

from typing import List, Set

import cv2
import numpy as np

from .fire_patterns import build_rotating_fire_phase_sets, extract_fire_cells_from_image
from .image_parsing import (
    build_object_matrix,
    build_teleport_pairs,
    build_wall_matrices,
    detect_colored_icons,
    find_single_cell,
    infer_grid_step,
    load_image_rgb,
    overlay_precise_cells,
)
from .models import (
    CONFUSION,
    FIRE_CENTER,
    GOAL,
    START,
    TELEPORT_TILES,
    Action,
    Cell,
    TurnResult,
)


class MazeEnvironment:
    def __init__(
        self,
        image_path: str,
        maze_size: int = 64,
    ):
        self.image_path = image_path
        self.maze_size = maze_size

        img_rgb = load_image_rgb(image_path)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        self.step_px, _, _ = infer_grid_step(gray)
        self.vertical_walls, self.horizontal_walls = build_wall_matrices(gray, self.step_px, n=maze_size)

        icons = detect_colored_icons(img_rgb, self.step_px, maze_size=maze_size)
        self.obj_matrix = build_object_matrix(icons, n=maze_size)
        overlay_precise_cells(
            self.obj_matrix,
            img_rgb,
            self.step_px,
            [CONFUSION, *TELEPORT_TILES, START, GOAL, FIRE_CENTER],
            maze_size=maze_size,
        )
        self.fire_center_cells = set(zip(*np.where(self.obj_matrix == FIRE_CENTER)))

        self.start = find_single_cell(self.obj_matrix, START, "start")
        self.goal = find_single_cell(self.obj_matrix, GOAL, "goal")
        self.teleport_pairs = build_teleport_pairs(self.obj_matrix)

        base_fire_cells = extract_fire_cells_from_image(image_path, self.step_px, maze_size=maze_size)
        self.fire_phase_sets = build_rotating_fire_phase_sets(base_fire_cells, maze_size, self.fire_center_cells)
        if not self.fire_phase_sets:
            self.fire_phase_sets = [set()]

        self.position: Cell = self.start
        self.turns_taken = 0
        self.deaths = 0
        self.confused_count = 0
        self.cells_visited: List[Cell] = []
        self.unique_cells: Set[Cell] = set()
        self.goal_reached = False

        self.confused_this_turn = False
        self.confused_turns_remaining = 0
        self.total_actions_executed = 0

    def reset(self) -> Cell:
        self.position = self.start
        self.turns_taken = 0
        self.deaths = 0
        self.confused_count = 0
        self.cells_visited = [self.start]
        self.unique_cells = {self.start}
        self.goal_reached = False
        self.confused_this_turn = False
        self.confused_turns_remaining = 0
        self.total_actions_executed = 0
        return self.start

    def get_active_fire_cells(self) -> Set[Cell]:
        phase = (self.total_actions_executed // 5) % len(self.fire_phase_sets)
        return self.fire_phase_sets[phase]

    def get_fire_phase(self) -> int:
        return (self.total_actions_executed // 5) % len(self.fire_phase_sets)

    def in_bounds(self, cell: Cell) -> bool:
        row, col = cell
        return 0 <= row < self.maze_size and 0 <= col < self.maze_size

    def can_move(self, a: Cell, b: Cell) -> bool:
        a_row, a_col = a
        b_row, b_col = b

        if not self.in_bounds(b):
            return False

        if b_row == a_row - 1 and b_col == a_col:
            return self.horizontal_walls[a_row, a_col] == 0
        if b_row == a_row + 1 and b_col == a_col:
            return self.horizontal_walls[a_row + 1, a_col] == 0
        if b_row == a_row and b_col == a_col - 1:
            return self.vertical_walls[a_row, a_col] == 0
        if b_row == a_row and b_col == a_col + 1:
            return self.vertical_walls[a_row, a_col + 1] == 0

        return False

    def apply_confusion(self, action: Action) -> Action:
        if action == Action.MOVE_UP:
            return Action.MOVE_DOWN
        if action == Action.MOVE_DOWN:
            return Action.MOVE_UP
        if action == Action.MOVE_LEFT:
            return Action.MOVE_RIGHT
        if action == Action.MOVE_RIGHT:
            return Action.MOVE_LEFT
        return Action.WAIT

    def action_to_target(self, pos: Cell, action: Action) -> Cell:
        row, col = pos
        if action == Action.MOVE_UP:
            return (row - 1, col)
        if action == Action.MOVE_DOWN:
            return (row + 1, col)
        if action == Action.MOVE_LEFT:
            return (row, col - 1)
        if action == Action.MOVE_RIGHT:
            return (row, col + 1)
        return pos

    def _apply_tile_effects(self, result: TurnResult) -> bool:
        if self.position in self.get_active_fire_cells():
            result.is_dead = True
            result.current_position = self.position
            self.deaths += 1
            return True

        tile = self.obj_matrix[self.position[0], self.position[1]]

        if tile == CONFUSION:
            self.confused_count += 1
            self.confused_turns_remaining = max(self.confused_turns_remaining, 2)
            self.confused_this_turn = True

        if self.position in self.teleport_pairs:
            self.position = self.teleport_pairs[self.position]
            result.teleported = True
            result.current_position = self.position

            if self.position in self.get_active_fire_cells():
                result.is_dead = True
                self.deaths += 1
                return True

        if self.position == self.goal or tile == GOAL:
            result.is_goal_reached = True
            self.goal_reached = True
            result.current_position = self.position
            return True

        return False

    def step_one_action(self, action: Action, turn_confused: bool) -> TurnResult:
        result = TurnResult(current_position=self.position)

        effective_action = self.apply_confusion(action) if turn_confused else action
        target = self.action_to_target(self.position, effective_action)

        if effective_action != Action.WAIT and not self.can_move(self.position, target):
            result.wall_hits = 1
            result.actions_executed = 1
            self.total_actions_executed += 1
            result.current_position = self.position
            result.is_confused = self.confused_this_turn
            return result

        self.position = target
        result.actions_executed = 1
        self.total_actions_executed += 1
        result.current_position = self.position
        self.cells_visited.append(self.position)
        self.unique_cells.add(self.position)

        self._apply_tile_effects(result)
        result.is_confused = turn_confused or self.confused_this_turn
        return result

    def finish_turn(self, result: TurnResult) -> TurnResult:
        self.turns_taken += 1

        if result.is_dead:
            self.position = self.start
            result.current_position = self.start

        if self.confused_turns_remaining > 0:
            self.confused_turns_remaining -= 1

        return result

    def step(self, actions: List[Action]) -> TurnResult:
        if not (1 <= len(actions) <= 5):
            raise ValueError("actions must contain between 1 and 5 actions")

        final_result = TurnResult(current_position=self.position)

        turn_confused = self.confused_turns_remaining > 0
        self.confused_this_turn = False

        for action in actions:
            atomic_result = self.step_one_action(action, turn_confused)

            final_result.wall_hits += atomic_result.wall_hits
            final_result.current_position = atomic_result.current_position
            final_result.is_dead = atomic_result.is_dead
            final_result.is_confused = atomic_result.is_confused
            final_result.is_goal_reached = atomic_result.is_goal_reached
            final_result.teleported = final_result.teleported or atomic_result.teleported
            final_result.actions_executed += atomic_result.actions_executed

            if atomic_result.is_dead or atomic_result.is_goal_reached:
                break

        return self.finish_turn(final_result)

    def get_episode_stats(self) -> dict:
        return {
            "turns_taken": self.turns_taken,
            "deaths": self.deaths,
            "confused": self.confused_count,
            "cells_explored": len(self.unique_cells),
            "goal_reached": self.goal_reached,
            "total_actions_executed": self.total_actions_executed,
        }
