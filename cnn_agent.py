from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from astar import astar_search, astar_search_debug
from environment import Action, MazeEnvironment, TurnResult

Cell = Tuple[int, int]

ACTIONS: List[Action] = [
    Action.MOVE_UP,
    Action.MOVE_DOWN,
    Action.MOVE_LEFT,
    Action.MOVE_RIGHT,
    Action.WAIT,
]
ACTION_TO_INDEX: Dict[Action, int] = {a: i for i, a in enumerate(ACTIONS)}

# ─────────────────────────────────────────────
# LOCAL GRID SPEC
# ─────────────────────────────────────────────
WINDOW = 11          # 11x11 cells centred on agent
HALF   = WINDOW // 2 # 5

# Channel indices in the local grid tensor (C, H, W)
CH_WALL_N   = 0   # wall on north edge of cell
CH_WALL_S   = 1
CH_WALL_W   = 2
CH_WALL_E   = 3
CH_HAZARD   = 4   # known death pit / fire
CH_TELEPORT = 5   # known teleport entry
CH_VISITED  = 6   # normalised visit count
CH_GOAL_DR  = 7   # (goal_row - cell_row) / maze_size  — broadcast
CH_GOAL_DC  = 8   # (goal_col - cell_col) / maze_size  — broadcast
CH_FIRE_PH  = 9   # fire phase progress (0-1)           — broadcast
CH_CONFUSED = 10  # confusion turns remaining (0-1)     — broadcast
NUM_CHANNELS = 11


# ─────────────────────────────────────────────
# WORLD MODEL  (same shape as original, kept
# minimal so you can swap in your teammate's)
# ─────────────────────────────────────────────
@dataclass
class WorldModel:
    known_walls:          Set[Tuple[Cell, Cell]]  = field(default_factory=set)
    known_hazards:        Set[Cell]                = field(default_factory=set)
    known_teleports:      Dict[Cell, Cell]         = field(default_factory=dict)
    visit_counts:         Dict[Cell, int]          = field(default_factory=dict)
    hazard_hit_counts:    Dict[Cell, int]          = field(default_factory=dict)
    action_transitions:   Dict[Tuple[Cell,int],Cell] = field(default_factory=dict)
    hazard_cooldown_until: Dict[Cell, int]         = field(default_factory=dict)


@dataclass
class AgentMemory:
    visited: Set[Cell] = field(default_factory=set)


# ─────────────────────────────────────────────
# REPLAY BUFFER
# ─────────────────────────────────────────────
@dataclass
class ReplayBuffer:
    capacity: int
    buffer: Deque = field(init=False)

    def __post_init__(self) -> None:
        self.buffer = deque(maxlen=self.capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s,  dtype=np.float32),
            np.array(a,  dtype=np.int64),
            np.array(r,  dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d,  dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ─────────────────────────────────────────────
# NETWORK
# ─────────────────────────────────────────────
class CNNQNetwork(nn.Module):
    """
    Small CNN over the 11x11 local grid, then MLP head.

    Input : (B, NUM_CHANNELS, WINDOW, WINDOW)
    Output: (B, num_actions)
    """

    def __init__(self, num_actions: int = 5) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            # (B, 11, 11, 11) -> (B, 32, 9, 9)
            nn.Conv2d(NUM_CHANNELS, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            # -> (B, 64, 7, 7)
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            # -> (B, 64, 5, 5)
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(),
        )
        # After 3 conv layers with kernel=3, no padding:
        # 11 -> 9 -> 7 -> 5  =>  64 * 5 * 5 = 1600
        conv_out = 64 * 5 * 5

        self.head = nn.Sequential(
            nn.Linear(conv_out, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        return self.head(self.conv(x).view(b, -1))


# ─────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────
class CNNAgent:
    """
    Pure RL agent using a CNN over a local grid window.

    Key design decisions
    --------------------
    * No A*. The network must learn navigation from scratch.
    * The world model exists ONLY to build the state tensor.
      It is never used to plan paths.
    * The agent is truly blind: walls, hazards, teleports are
      discovered only through feedback in TurnResult.
    * Fire phase and confusion are embedded in the state so
      the network can learn timing and control inversion.
    """

    def __init__(
        self,
        env:                    Optional[MazeEnvironment] = None,
        gamma:                  float = 0.99,
        lr:                     float = 5e-4,
        batch_size:             int   = 256,
        replay_capacity:        int   = 200_000,
        min_replay_size:        int   = 5_000,
        target_update_interval: int   = 1_000,
        epsilon_start:          float = 1.0,
        epsilon_end:            float = 0.05,
        epsilon_decay_steps:    int   = 150_000,
        astar_follow_prob:      float = 0.80,
        device:                 Optional[str] = None,
    ) -> None:

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.gamma                  = gamma
        self.batch_size             = batch_size
        self.min_replay_size        = min_replay_size
        self.target_update_interval = target_update_interval
        self.epsilon_start          = epsilon_start
        self.epsilon_end            = epsilon_end
        self.epsilon_decay_steps    = epsilon_decay_steps
        self.astar_follow_prob      = astar_follow_prob

        self.policy_net = CNNQNetwork(len(ACTIONS)).to(self.device)
        self.target_net = CNNQNetwork(len(ACTIONS)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer    = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay       = ReplayBuffer(replay_capacity)
        self.training_steps = 0

        # Per-episode state
        self.env:               Optional[MazeEnvironment] = None
        self.world_models:      Dict[str, WorldModel]     = {}
        self.active_map_key:    Optional[str]             = None
        self._wm:               Optional[WorldModel]      = None  # active world model

        self.episode_steps:     int                       = 0
        self.last_result:       Optional[TurnResult]      = None
        self.last_action_idx:   int                       = ACTION_TO_INDEX[Action.WAIT]
        self.pending_pos:       Optional[Cell]            = None
        self.pending_action:    Optional[int]             = None
        self.pending_confused:  bool                      = False
        self.memory:            AgentMemory               = AgentMemory()

        # Keep these attributes for compatibility with visualizer overlays.
        self.current_path: List[Cell] = []
        self.last_search_expanded: List[Cell] = []
        self.last_search_closed: Set[Cell] = set()

        if env is not None:
            self.bind_environment(env)

    # ──────────────────────────────────────────
    # ENVIRONMENT BINDING
    # ──────────────────────────────────────────
    def _map_key(self, env: MazeEnvironment) -> str:
        p = getattr(env, "image_path", "")
        return str(Path(p).resolve()) if p else f"maze_{env.maze_size}"

    def bind_environment(self, env: MazeEnvironment) -> None:
        self.env = env
        self.active_map_key = self._map_key(env)
        if self.active_map_key not in self.world_models:
            self.world_models[self.active_map_key] = WorldModel()
        self._wm = self.world_models[self.active_map_key]
        self._init_episode()

    def reset_episode(self) -> None:
        if self.env is not None:
            self._init_episode()

    def _init_episode(self) -> None:
        self.episode_steps   = 0
        self.last_result     = None
        self.last_action_idx = ACTION_TO_INDEX[Action.WAIT]
        self.pending_pos     = None
        self.pending_action  = None
        self.pending_confused = False
        self.current_path = []
        self.last_search_expanded = []
        self.last_search_closed = set()
        self.memory.visited.clear()
        # Visit the start cell
        pos = self.env.position
        wm = self._wm
        wm.visit_counts[pos] = wm.visit_counts.get(pos, 0) + 1
        self.memory.visited.add(pos)

    # ──────────────────────────────────────────
    # WORLD MODEL UPDATE
    # ──────────────────────────────────────────
    @staticmethod
    def _edge_key(a: Cell, b: Cell) -> Tuple[Cell, Cell]:
        return (a, b) if a <= b else (b, a)

    def _update_world_model(
        self,
        prev_pos:    Cell,
        action_idx:  int,
        result:      TurnResult,
        was_confused: bool,
    ) -> None:
        wm     = self._wm
        env    = self.env
        action = ACTIONS[action_idx]
        current_step = int(env.total_actions_executed)

        # Confusion flips the effective action
        effective = env.apply_confusion(action) if was_confused else action

        # Wall detection
        if result.wall_hits > 0 and result.current_position == prev_pos:
            target = env.action_to_target(prev_pos, effective)
            if env.in_bounds(target):
                wm.known_walls.add(self._edge_key(prev_pos, target))

        # Hazard detection (death and not a teleport outcome)
        if result.is_dead and not result.teleported:
            target = env.action_to_target(prev_pos, effective)
            if env.in_bounds(target):
                wm.known_hazards.add(target)
                wm.hazard_hit_counts[target] = wm.hazard_hit_counts.get(target, 0) + 1
                wm.hazard_cooldown_until[target] = max(
                    wm.hazard_cooldown_until.get(target, 0),
                    current_step + 5,
                )

        # Teleport mapping
        if result.teleported and not result.is_dead:
            target = env.action_to_target(prev_pos, effective)
            if env.in_bounds(target) and target != result.current_position:
                wm.known_teleports[target] = result.current_position

        # Successful move — record transition
        if result.wall_hits == 0 and not result.is_dead:
            key = (prev_pos, ACTION_TO_INDEX[effective])
            wm.action_transitions[key] = result.current_position

        if result.wall_hits == 0 and not result.is_dead and result.current_position in wm.known_hazards:
            wm.hazard_cooldown_until[result.current_position] = max(
                wm.hazard_cooldown_until.get(result.current_position, 0),
                current_step + 5,
            )

        # Visit count
        pos = result.current_position
        wm.visit_counts[pos] = wm.visit_counts.get(pos, 0) + 1
        self.memory.visited.add(pos)

    @staticmethod
    def _is_move_action(action: Action) -> bool:
        return action in {
            Action.MOVE_UP,
            Action.MOVE_DOWN,
            Action.MOVE_LEFT,
            Action.MOVE_RIGHT,
        }

    def _predicted_next_cell(self, env: MazeEnvironment, cell: Cell, action: Action) -> Optional[Cell]:
        if not self._is_move_action(action):
            return cell

        target = env.action_to_target(cell, action)
        if not env.in_bounds(target):
            return None

        if self._edge_key(cell, target) in self._wm.known_walls:
            return None

        return self._wm.action_transitions.get((cell, ACTION_TO_INDEX[action]), target)

    def _find_action_to_next(self, env: MazeEnvironment, cell: Cell, next_cell: Cell) -> Optional[Action]:
        for action in (Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT):
            predicted = self._predicted_next_cell(env, cell, action)
            if predicted == next_cell:
                return action
        return None

    def _is_hazard_blocked_now(self, cell: Cell, env: MazeEnvironment) -> bool:
        cutoff = self._wm.hazard_cooldown_until.get(cell, 0)
        if self.episode_steps < cutoff:
            return True

        active_fire = env.get_active_fire_cells() if hasattr(env, "get_active_fire_cells") else set()
        return cell in active_fire

    def _neighbors_ignoring_hazard_cooldown(self, env: MazeEnvironment, cell: Cell) -> List[Cell]:
        out: List[Cell] = []
        seen: Set[Cell] = set()
        for action in (Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT):
            predicted = self._predicted_next_cell(env, cell, action)
            if predicted is None:
                continue
            effective = self._wm.known_teleports.get(predicted, predicted)
            if effective not in seen:
                seen.add(effective)
                out.append(effective)
        return out

    def _wait_for_hazard_action(self, env: MazeEnvironment) -> Optional[int]:
        pos = env.position
        for action in (Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT):
            target = env.action_to_target(pos, action)
            if not env.in_bounds(target):
                continue
            if target in self._wm.known_hazards and self._is_hazard_blocked_now(target, env):
                path = astar_search(
                    pos,
                    env.goal,
                    lambda cell: self._neighbors_ignoring_hazard_cooldown(env, cell),
                )
                if len(path) >= 2 and path[1] == target:
                    return ACTION_TO_INDEX[Action.WAIT]
        return None

    def _neighbors_for_astar(self, env: MazeEnvironment, cell: Cell) -> List[Cell]:
        out: List[Cell] = []
        seen: Set[Cell] = set()

        for action in (Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT):
            predicted = self._predicted_next_cell(env, cell, action)
            if predicted is None:
                continue

            if predicted in self._wm.known_hazards and self._is_hazard_blocked_now(predicted, env):
                continue

            effective = self._wm.known_teleports.get(predicted, predicted)
            if effective not in seen:
                seen.add(effective)
                out.append(effective)

        return out

    def _astar_action_index(
        self,
        env: MazeEnvironment,
        pre_step_confused: bool = False,
        debug: bool = False,
    ) -> Optional[int]:
        if env.position == env.goal:
            self.current_path = [env.position]
            return ACTION_TO_INDEX[Action.WAIT]

        if debug:
            debug_out = astar_search_debug(
                env.position,
                env.goal,
                lambda cell: self._neighbors_for_astar(env, cell),
            )
            self.current_path = debug_out["path"]
            self.last_search_expanded = debug_out["expanded_order"]
            self.last_search_closed = debug_out["closed_set"]
            path = debug_out["path"]
        else:
            path = astar_search(
                env.position,
                env.goal,
                lambda cell: self._neighbors_for_astar(env, cell),
            )
            self.current_path = path

        if len(path) < 2:
            wait_idx = self._wait_for_hazard_action(env)
            if wait_idx is not None:
                return wait_idx
            return None

        next_cell = path[1]
        teleport_entry = None
        for entry, dest in self._wm.known_teleports.items():
            if dest == next_cell:
                teleport_entry = entry
                break

        target_cell = teleport_entry if teleport_entry is not None else next_cell
        desired_action = self._find_action_to_next(env, path[0], target_cell)
        if desired_action is None:
            return None

        if pre_step_confused:
            return ACTION_TO_INDEX[env.apply_confusion(desired_action)]

        return ACTION_TO_INDEX[desired_action]

    # ──────────────────────────────────────────
    # STATE EXTRACTION  →  (C, H, W) tensor
    # ──────────────────────────────────────────
    def extract_state(self, env: MazeEnvironment) -> np.ndarray:
        """
        Build an (NUM_CHANNELS, WINDOW, WINDOW) float32 array.

        The agent sits at the centre (HALF, HALF).
        Out-of-bounds cells are treated as walls on all sides.
        """
        wm  = self._wm
        pos = env.position
        pr, pc = pos
        n   = env.maze_size

        grid = np.zeros((NUM_CHANNELS, WINDOW, WINDOW), dtype=np.float32)

        # Fire phase progress  (0 = just rotated, 1 = about to rotate)
        fire_phase_progress = (env.total_actions_executed % 5) / 5.0

        # Scalar channels broadcast to the whole window
        grid[CH_FIRE_PH]  = fire_phase_progress
        grid[CH_CONFUSED] = min(1.0, env.confused_turns_remaining / 2.0)

        for wr in range(WINDOW):
            for wc in range(WINDOW):
                cr = pr + (wr - HALF)
                cc = pc + (wc - HALF)

                # Out-of-bounds: mark all walls, nothing else
                if not (0 <= cr < n and 0 <= cc < n):
                    grid[CH_WALL_N, wr, wc] = 1.0
                    grid[CH_WALL_S, wr, wc] = 1.0
                    grid[CH_WALL_W, wr, wc] = 1.0
                    grid[CH_WALL_E, wr, wc] = 1.0
                    continue

                cell = (cr, cc)

                # ── Walls (discovered) ──────────────────
                north = (cr - 1, cc)
                south = (cr + 1, cc)
                west  = (cr, cc - 1)
                east  = (cr, cc + 1)

                if self._edge_key(cell, north) in wm.known_walls or cr == 0:
                    grid[CH_WALL_N, wr, wc] = 1.0
                if self._edge_key(cell, south) in wm.known_walls or cr == n - 1:
                    grid[CH_WALL_S, wr, wc] = 1.0
                if self._edge_key(cell, west)  in wm.known_walls or cc == 0:
                    grid[CH_WALL_W, wr, wc] = 1.0
                if self._edge_key(cell, east)  in wm.known_walls or cc == n - 1:
                    grid[CH_WALL_E, wr, wc] = 1.0

                # ── Hazard ──────────────────────────────
                # Dynamic fire: only mark as hazard if we know it AND
                # it's currently in the active fire phase set.
                # This teaches the network fire is temporal.
                if cell in wm.known_hazards:
                    active_fire = env.get_active_fire_cells() if hasattr(env, "get_active_fire_cells") else set()
                    grid[CH_HAZARD, wr, wc] = 1.0 if cell in active_fire else 0.5
                    # 1.0 = active right now, 0.5 = known hazard but currently clear

                # ── Teleport ────────────────────────────
                if cell in wm.known_teleports:
                    grid[CH_TELEPORT, wr, wc] = 1.0

                # ── Visit count (normalised) ─────────────
                grid[CH_VISITED, wr, wc] = min(1.0, wm.visit_counts.get(cell, 0) / 20.0)

                # ── Goal direction (relative, per cell) ──
                gr, gc = env.goal
                grid[CH_GOAL_DR, wr, wc] = (gr - cr) / float(n)
                grid[CH_GOAL_DC, wr, wc] = (gc - cc) / float(n)

        return grid

    # ──────────────────────────────────────────
    # EPSILON
    # ──────────────────────────────────────────
    def epsilon(self) -> float:
        ratio = min(1.0, self.training_steps / max(1.0, float(self.epsilon_decay_steps)))
        return self.epsilon_start + ratio * (self.epsilon_end - self.epsilon_start)

    # ──────────────────────────────────────────
    # ACTION SELECTION
    # ──────────────────────────────────────────
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and random.random() < self.epsilon():
            return random.randrange(len(ACTIONS))

        with torch.no_grad():
            t = torch.from_numpy(state).unsqueeze(0).to(self.device)
            return int(torch.argmax(self.policy_net(t), dim=1).item())

    # ──────────────────────────────────────────
    # REWARD
    # ──────────────────────────────────────────
    def _compute_reward(
        self,
        prev_pos:           Cell,
        result:             TurnResult,
        discovered_new:     bool,
        warmup:             bool,
    ) -> float:
        wm      = self._wm
        env     = self.env
        new_pos = result.current_position

        prev_dist = abs(prev_pos[0] - env.goal[0]) + abs(prev_pos[1] - env.goal[1])
        new_dist  = abs(new_pos[0]  - env.goal[0]) + abs(new_pos[1]  - env.goal[1])

        r = -0.05                               # small step cost
        r += 0.50 * (prev_dist - new_dist)      # progress toward goal

        if result.wall_hits > 0:
            r -= 1.0 * result.wall_hits         # wall bumping

        if result.is_dead:
            r -= 30.0                           # death — heavy penalty

        if result.is_goal_reached:
            r += 50.0 if warmup else 200.0      # goal — big bonus

        if discovered_new:
            r += 1.0 if warmup else 0.30        # exploration bonus

        if result.teleported and not result.is_dead:
            known_before = prev_pos in wm.known_teleports
            r += 0.5 if not known_before else 0.15   # reward discovering teleports

        # Penalise staying still unless we just discovered something useful
        if result.current_position == prev_pos and not result.is_goal_reached:
            if not discovered_new:
                r -= 0.20

        return float(r)

    # ──────────────────────────────────────────
    # OPTIMISE
    # ──────────────────────────────────────────
    def optimize_step(self) -> Optional[float]:
        if len(self.replay) < max(self.batch_size, self.min_replay_size):
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        s  = torch.from_numpy(states).to(self.device)
        a  = torch.from_numpy(actions).to(self.device)
        r  = torch.from_numpy(rewards).to(self.device)
        ns = torch.from_numpy(next_states).to(self.device)
        d  = torch.from_numpy(dones).to(self.device)

        # Double DQN
        q_vals = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_actions = torch.argmax(self.policy_net(ns), dim=1, keepdim=True)
            next_q       = self.target_net(ns).gather(1, best_actions).squeeze(1)
            targets      = r + (1.0 - d) * self.gamma * next_q

        loss = nn.functional.smooth_l1_loss(q_vals, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    # ──────────────────────────────────────────
    # TRAINING LOOP
    # ──────────────────────────────────────────
    def train(
        self,
        map_paths:         List[str],
        episodes:          int,
        max_turns:         int        = 10_000,
        warmup_episodes:   int        = 30,
        log_every:         int        = 25,
        checkpoint_path:   Optional[str] = None,
        checkpoint_every:  int        = 50,
        seed:              int        = 42,
    ) -> Dict:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        history = {
            "episode_reward": [],
            "episode_loss":   [],
            "episode_steps":  [],
            "success":        [],
            "deaths":         [],
        }

        for ep in range(1, episodes + 1):
            map_path = random.choice(map_paths)
            env = MazeEnvironment(image_path=map_path, maze_size=64)
            env.reset()
            self.bind_environment(env)

            warmup   = ep <= warmup_episodes
            ep_reward = 0.0
            ep_loss   = 0.0
            loss_n    = 0

            for _ in range(max_turns):
                state           = self.extract_state(env)
                was_confused    = env.confused_turns_remaining > 0
                astar_idx       = self._astar_action_index(env, pre_step_confused=was_confused, debug=False)
                if astar_idx is not None and (ep > warmup_episodes or random.random() < self.astar_follow_prob):
                    action_idx = astar_idx
                else:
                    action_idx = self.select_action(state, explore=True)
                prev_pos        = env.position
                discovered_new  = env.position not in self._wm.visit_counts

                result = env.step([ACTIONS[action_idx]])

                self._update_world_model(prev_pos, action_idx, result, was_confused)
                self.last_result     = result
                self.last_action_idx = action_idx
                self.episode_steps  += 1

                next_state = self.extract_state(env)
                done       = result.is_goal_reached

                reward = self._compute_reward(prev_pos, result, discovered_new, warmup)
                ep_reward += reward

                self.replay.push(state, action_idx, reward, next_state, done)

                loss = self.optimize_step()
                if loss is not None:
                    ep_loss += loss
                    loss_n  += 1

                if done:
                    break

            history["episode_reward"].append(ep_reward)
            history["episode_loss"].append(ep_loss / max(1, loss_n))
            history["episode_steps"].append(float(env.turns_taken))
            history["success"].append(float(env.goal_reached))
            history["deaths"].append(float(env.deaths))

            if ep % log_every == 0:
                sl   = slice(max(0, ep - log_every), ep)
                sr   = float(np.mean(history["success"][sl]))
                ar   = float(np.mean(history["episode_reward"][sl]))
                ast  = float(np.mean(history["episode_steps"][sl]))
                adth = float(np.mean(history["deaths"][sl]))
                print(
                    f"[cnn] ep={ep:04d} "
                    f"phase={'warmup' if warmup else 'train'} "
                    f"eps={self.epsilon():.3f} "
                    f"success={sr:.2f} "
                    f"avg_reward={ar:.1f} "
                    f"avg_turns={ast:.1f} "
                    f"avg_deaths={adth:.1f}"
                )

            if checkpoint_path and ep % checkpoint_every == 0:
                p = Path(checkpoint_path)
                snap = p.with_name(f"{p.stem}_ep{ep:04d}{p.suffix}")
                self.save(str(snap))
                print(f"[cnn] checkpoint -> {snap}")

        self.target_net.load_state_dict(self.policy_net.state_dict())
        return history

    # ──────────────────────────────────────────
    # INFERENCE  (plan_turn interface)
    # ──────────────────────────────────────────
    def plan_turn(self, last_result: Optional[TurnResult]) -> List[Action]:
        """
        Drop-in replacement for the original agent's plan_turn.
        Returns a list of 1 action (extend to multi-action sequence later).
        """
        if self.env is None:
            raise RuntimeError("Agent not bound to environment")

        if last_result is not None and self.pending_pos is not None:
            self._update_world_model(
                self.pending_pos,
                self.pending_action,
                last_result,
                self.pending_confused,
            )
            self.last_result     = last_result
            self.last_action_idx = self.pending_action
            self.episode_steps  += 1

        if self.env.position == self.env.goal:
            return [Action.WAIT]

        state      = self.extract_state(self.env)
        pre_step_confused = self.env.confused_turns_remaining > 0
        astar_idx = self._astar_action_index(self.env, pre_step_confused=pre_step_confused, debug=True)
        if astar_idx is not None:
            action_idx = astar_idx
        else:
            action_idx = self.select_action(state, explore=False)

        self.pending_pos     = self.env.position
        self.pending_action  = action_idx
        self.pending_confused = self.env.confused_turns_remaining > 0

        return [ACTIONS[action_idx]]

    # ──────────────────────────────────────────
    # SAVE / LOAD
    # ──────────────────────────────────────────
    def save(self, path: str, map_paths: Optional[List[str]] = None) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                # DQN-compatible key names
                "policy_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_steps": self.training_steps,
                "gamma": self.gamma,
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "batch_size": self.batch_size,
                "target_update_interval": self.target_update_interval,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay_steps": self.epsilon_decay_steps,
                "astar_follow_prob": self.astar_follow_prob,
                "min_replay_size": self.min_replay_size,
                "replay_capacity": self.replay.capacity,
                "world_models": self.world_models,
                "map_paths": list(map_paths or []),
                # CNN-specific metadata
                "agent_type": "cnn",
            },
            str(p),
        )

    @classmethod
    def load(
        cls,
        path:   str,
        env:    Optional[MazeEnvironment] = None,
        device: Optional[str] = None,
    ) -> "CNNAgent":
        try:
            ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=device or "cpu")

        hp = ckpt.get("hparams", {})
        lr = float(
            ckpt.get(
                "learning_rate",
                hp.get("learning_rate", 5e-4),
            )
        )

        agent = cls(
            env                    = None,
            gamma                  = float(ckpt.get("gamma", hp.get("gamma", 0.99))),
            lr                     = lr,
            batch_size             = int(ckpt.get("batch_size", hp.get("batch_size", 256))),
            replay_capacity        = int(ckpt.get("replay_capacity", 200_000)),
            min_replay_size        = int(ckpt.get("min_replay_size", hp.get("min_replay_size", 5_000))),
            target_update_interval = int(ckpt.get("target_update_interval", hp.get("target_update_interval", 1_000))),
            epsilon_start          = float(ckpt.get("epsilon_start", hp.get("epsilon_start", 1.0))),
            epsilon_end            = float(ckpt.get("epsilon_end", hp.get("epsilon_end", 0.05))),
            epsilon_decay_steps    = int(ckpt.get("epsilon_decay_steps", hp.get("epsilon_decay_steps", 150_000))),
            astar_follow_prob      = float(ckpt.get("astar_follow_prob", hp.get("astar_follow_prob", 0.80))),
            device                 = device,
        )

        policy = ckpt.get("policy_state_dict", ckpt.get("policy"))
        target = ckpt.get("target_state_dict", ckpt.get("target", policy))
        optimizer_state = ckpt.get("optimizer_state_dict", ckpt.get("optimizer"))

        if policy is None:
            raise ValueError("Checkpoint missing policy network state")

        agent.policy_net.load_state_dict(policy)
        if target is not None:
            agent.target_net.load_state_dict(target)
        else:
            agent.target_net.load_state_dict(policy)
        if optimizer_state is not None:
            agent.optimizer.load_state_dict(optimizer_state)
        agent.training_steps = int(ckpt.get("training_steps", 0))

        # Restore world models
        raw_wms = ckpt.get("world_models", {})
        for k, v in raw_wms.items():
            if isinstance(v, WorldModel):
                agent.world_models[k] = v
            elif isinstance(v, dict):
                agent.world_models[k] = WorldModel(
                    known_walls        = set(map(tuple, v.get("known_walls", []))),
                    known_hazards      = set(map(tuple, v.get("known_hazards", []))),
                    known_teleports    = {tuple(a): tuple(b) for a, b in v.get("known_teleports", {}).items()},
                    visit_counts       = {tuple(k2): v2 for k2, v2 in v.get("visit_counts", {}).items()},
                    hazard_hit_counts  = {tuple(k2): v2 for k2, v2 in v.get("hazard_hit_counts", {}).items()},
                    action_transitions = {(tuple(k2[0]), k2[1]): tuple(v2) for k2, v2 in v.get("action_transitions", {}).items()},
                )

        if env is not None:
            agent.bind_environment(env)

        return agent

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        env: Optional[MazeEnvironment] = None,
        device: Optional[str] = None,
    ) -> "CNNAgent":
        return cls.load(checkpoint_path, env=env, device=device)

    def evaluate(
        self,
        map_paths: List[str],
        episodes_per_map: int,
        max_turns: int,
    ) -> Dict[str, Dict[str, float]]:
        if episodes_per_map <= 0:
            raise ValueError("episodes_per_map must be >= 1")

        metrics: Dict[str, Dict[str, float]] = {}
        self.policy_net.eval()

        for map_path in map_paths:
            successes = 0
            steps_list: List[int] = []
            deaths_list: List[int] = []

            for _ in range(episodes_per_map):
                env = MazeEnvironment(image_path=map_path, maze_size=64)
                env.reset()
                self.bind_environment(env)

                for _ in range(max_turns):
                    state = self.extract_state(env)
                    pre_step_confused = env.confused_turns_remaining > 0
                    astar_idx = self._astar_action_index(env, pre_step_confused=pre_step_confused, debug=False)
                    if astar_idx is not None:
                        action_idx = astar_idx
                    else:
                        action_idx = self.select_action(state, explore=False)

                    prev_pos = env.position
                    was_confused = env.confused_turns_remaining > 0
                    result = env.step([ACTIONS[action_idx]])

                    self._update_world_model(prev_pos, action_idx, result, was_confused)
                    self.last_result = result
                    self.last_action_idx = action_idx
                    self.episode_steps += 1

                    if result.is_goal_reached:
                        break

                stats = env.get_episode_stats()
                successes += int(stats["goal_reached"])
                steps_list.append(int(stats["turns_taken"]))
                deaths_list.append(int(stats["deaths"]))

            metrics[map_path] = {
                "success_rate": successes / float(episodes_per_map),
                "avg_steps": float(np.mean(steps_list)) if steps_list else 0.0,
                "avg_deaths": float(np.mean(deaths_list)) if deaths_list else 0.0,
            }

        self.policy_net.train()
        return metrics