from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

Cell = Tuple[int, int]
ACTIONS_PER_TURN = 5

EMPTY = 0
FIRE = 1
CONFUSION = 2
TP_PURPLE = 3
TP_RED = 4
TP_GREEN = 5
TP_LAVENDER = 6
START = 7
GOAL = 8
FIRE_CENTER = 9
UNKNOWN = 99

TELEPORT_TILES = (TP_PURPLE, TP_RED, TP_GREEN, TP_LAVENDER)

TARGET_COLORS = {
    FIRE: [(255, 145, 76)],
    CONFUSION: [(255, 222, 89)],
    TP_PURPLE: [(140, 82, 255)],
    TP_RED: [(255, 49, 50)],
    TP_GREEN: [(1, 191, 99), (42, 193, 126)],
    TP_LAVENDER: [(226, 169, 241)],
    START: [(15, 192, 223)],
    GOAL: [(0, 74, 173)],
    FIRE_CENTER: [(253, 183, 140)],
}

COLOR_TOL = 45


class Action(Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    WAIT = 4


@dataclass
class TurnResult:
    wall_hits: int = 0
    current_position: Cell = (0, 0)
    is_dead: bool = False
    is_confused: bool = False
    is_goal_reached: bool = False
    teleported: bool = False
    actions_executed: int = 0


@dataclass
class Icon:
    x: int
    y: int
    w: int
    h: int
    cx: float
    cy: float
    row: int
    col: int
    kind: int
    rgb_mean: Tuple[float, float, float]
