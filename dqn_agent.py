from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment import Action, MazeEnvironment, TurnResult

Cell = Tuple[int, int]

ACTIONS: List[Action] = [
    Action.MOVE_UP,
    Action.MOVE_DOWN,
    Action.MOVE_LEFT,
    Action.MOVE_RIGHT,
    Action.WAIT,
]
ACTION_TO_INDEX: Dict[Action, int] = {action: idx for idx, action in enumerate(ACTIONS)}


@dataclass
class ReplayBuffer:
    capacity: int
    buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = field(init=False)

    def __post_init__(self) -> None:
        self.buffer = deque(maxlen=self.capacity)

    def push(
        self,
        state: np.ndarray,
        action_idx: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action_idx, reward, next_state, float(done)))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class AgentMemory:
    visited: Set[Cell] = field(default_factory=set)


class DQNAgent:
    """
    DQN agent that supports:
    1) offline training over one or more maze images
    2) greedy inference via plan_turn for the existing visualizer loop
    """

    input_dim: int = 20

    def __init__(
        self,
        env: Optional[MazeEnvironment] = None,
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        replay_capacity: int = 120_000,
        min_replay_size: int = 2_000,
        target_update_interval: int = 500,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 75_000,
        device: Optional[str] = None,
    ) -> None:
        self.env: Optional[MazeEnvironment] = env

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.min_replay_size = min_replay_size
        self.target_update_interval = target_update_interval

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        self.policy_net = QNetwork(self.input_dim, len(ACTIONS)).to(self.device)
        self.target_net = QNetwork(self.input_dim, len(ACTIONS)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)

        self.training_steps = 0
        self.memory = AgentMemory()
        self.visit_counts: Dict[Cell, int] = {}
        self.last_result: Optional[TurnResult] = None
        self.last_action_idx: int = ACTION_TO_INDEX[Action.WAIT]

        # Keep these attributes to stay compatible with the visualizer overlays.
        self.last_search_expanded: List[Cell] = []
        self.last_search_closed: Set[Cell] = set()

        if self.env is not None:
            self._init_episode_state(self.env)

    def bind_environment(self, env: MazeEnvironment) -> None:
        self.env = env
        self._init_episode_state(env)

    def reset_episode(self) -> None:
        if self.env is not None:
            self._init_episode_state(self.env)

    def _init_episode_state(self, env: MazeEnvironment) -> None:
        self.memory.visited.clear()
        self.memory.visited.add(env.position)
        self.visit_counts = {env.position: 1}
        self.last_result = None
        self.last_action_idx = ACTION_TO_INDEX[Action.WAIT]

    @staticmethod
    def _manhattan(a: Cell, b: Cell) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _normalize_dist(self, env: MazeEnvironment, dist: float) -> float:
        return float(dist) / max(1.0, float(2 * env.maze_size))

    def _record_step(self, action_idx: int, result: TurnResult) -> None:
        self.last_action_idx = action_idx
        self.last_result = result
        self.memory.visited.add(result.current_position)
        self.visit_counts[result.current_position] = self.visit_counts.get(result.current_position, 0) + 1

    def extract_state(self, env: MazeEnvironment) -> np.ndarray:
        pos = env.position
        goal = env.goal

        dr = float(goal[0] - pos[0])
        dc = float(goal[1] - pos[1])
        current_dist = self._manhattan(pos, goal)

        if self.last_result is None:
            wall_hits_norm = 0.0
            dead_flag = 0.0
            goal_flag = 0.0
            teleported_flag = 0.0
            actions_executed_norm = 0.0
            was_confused_flag = 0.0
        else:
            wall_hits_norm = min(1.0, self.last_result.wall_hits / 5.0)
            dead_flag = float(self.last_result.is_dead)
            goal_flag = float(self.last_result.is_goal_reached)
            teleported_flag = float(self.last_result.teleported)
            actions_executed_norm = min(1.0, self.last_result.actions_executed / 5.0)
            was_confused_flag = float(self.last_result.is_confused)

        last_action_one_hot = [0.0] * len(ACTIONS)
        last_action_one_hot[self.last_action_idx] = 1.0

        visit_count_norm = min(1.0, self.visit_counts.get(pos, 0) / 10.0)

        features: List[float] = [
            pos[0] / max(1.0, float(env.maze_size - 1)),
            pos[1] / max(1.0, float(env.maze_size - 1)),
            dr / max(1.0, float(env.maze_size)),
            dc / max(1.0, float(env.maze_size)),
            self._normalize_dist(env, current_dist),
            float(env.confused_turns_remaining > 0),
            min(1.0, env.confused_turns_remaining / 2.0),
            visit_count_norm,
            wall_hits_norm,
            dead_flag,
            goal_flag,
            teleported_flag,
            actions_executed_norm,
            was_confused_flag,
            1.0,
        ]
        features.extend(last_action_one_hot)

        return np.asarray(features, dtype=np.float32)

    def epsilon(self) -> float:
        ratio = min(1.0, self.training_steps / max(1.0, float(self.epsilon_decay_steps)))
        return self.epsilon_start + ratio * (self.epsilon_end - self.epsilon_start)

    def select_action_index(self, state: np.ndarray, explore: bool) -> int:
        if explore and random.random() < self.epsilon():
            return random.randrange(len(ACTIONS))

        with torch.no_grad():
            state_tensor = torch.from_numpy(state).to(self.device).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def _compute_reward(self, prev_pos: Cell, result: TurnResult, env: MazeEnvironment) -> float:
        new_pos = result.current_position

        prev_dist = self._manhattan(prev_pos, env.goal)
        new_dist = self._manhattan(new_pos, env.goal)

        reward = -0.10
        reward += 0.40 * (prev_dist - new_dist)

        if result.wall_hits > 0:
            reward -= 1.5 * result.wall_hits
        if result.is_dead:
            reward -= 25.0
        if result.is_goal_reached:
            reward += 200.0
        if result.is_confused:
            reward -= 0.05
        if result.current_position == prev_pos and not result.is_goal_reached:
            reward -= 0.15

        return float(reward)

    def optimize_step(self) -> Optional[float]:
        if len(self.replay_buffer) < max(self.batch_size, self.min_replay_size):
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = torch.argmax(self.policy_net(next_states_t), dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            targets = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        loss = nn.functional.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        if self.training_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def train(
        self,
        map_paths: Sequence[str],
        episodes: int,
        max_turns: int,
        log_every: int = 25,
        seed: int = 7,
        checkpoint_path: Optional[str] = None,
        checkpoint_every: int = 25,
    ) -> Dict[str, List[float]]:
        if not map_paths:
            raise ValueError("map_paths cannot be empty")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        history: Dict[str, List[float]] = {
            "episode_reward": [],
            "episode_loss": [],
            "episode_steps": [],
            "success": [],
        }

        for episode_idx in range(1, episodes + 1):
            map_path = random.choice(map_paths)
            env = MazeEnvironment(image_path=map_path, maze_size=64)
            env.reset()
            self.bind_environment(env)

            total_reward = 0.0
            total_loss = 0.0
            loss_count = 0

            for _ in range(max_turns):
                state = self.extract_state(env)
                action_idx = self.select_action_index(state, explore=True)
                action = ACTIONS[action_idx]

                prev_pos = env.position
                result = env.step([action])
                self._record_step(action_idx, result)

                next_state = self.extract_state(env)
                done = result.is_goal_reached or env.turns_taken >= max_turns
                reward = self._compute_reward(prev_pos, result, env)

                total_reward += reward
                self.replay_buffer.push(state, action_idx, reward, next_state, done)

                self.training_steps += 1
                loss = self.optimize_step()
                if loss is not None:
                    total_loss += loss
                    loss_count += 1

                if done:
                    break

            mean_loss = total_loss / max(1, loss_count)
            history["episode_reward"].append(float(total_reward))
            history["episode_loss"].append(float(mean_loss))
            history["episode_steps"].append(float(env.turns_taken))
            history["success"].append(float(env.goal_reached))

            if episode_idx % max(1, log_every) == 0:
                recent = slice(max(0, episode_idx - log_every), episode_idx)
                success_rate = float(np.mean(history["success"][recent]))
                avg_reward = float(np.mean(history["episode_reward"][recent]))
                avg_steps = float(np.mean(history["episode_steps"][recent]))
                print(
                    f"[train] ep={episode_idx:04d} "
                    f"eps={self.epsilon():.3f} "
                    f"success={success_rate:.2f} "
                    f"avg_reward={avg_reward:.2f} "
                    f"avg_steps={avg_steps:.1f}"
                )

            if checkpoint_path and episode_idx % max(1, checkpoint_every) == 0:
                ckpt_target = Path(checkpoint_path)
                snapshot = ckpt_target.with_name(f"{ckpt_target.stem}_ep{episode_idx:04d}{ckpt_target.suffix}")
                self.save(str(snapshot), map_paths=map_paths)
                print(f"[train] checkpoint saved -> {snapshot}")

        self.target_net.load_state_dict(self.policy_net.state_dict())
        return history

    def evaluate(
        self,
        map_paths: Sequence[str],
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
                    action_idx = self.select_action_index(state, explore=False)
                    result = env.step([ACTIONS[action_idx]])
                    self._record_step(action_idx, result)
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

    def plan_turn(self, last_result: Optional[TurnResult]) -> List[Action]:
        if self.env is None:
            raise RuntimeError("DQNAgent is not bound to an environment")

        if last_result is not None:
            self.last_result = last_result
            self.memory.visited.add(last_result.current_position)
            self.visit_counts[last_result.current_position] = self.visit_counts.get(last_result.current_position, 0) + 1

        if self.env.position == self.env.goal:
            return [Action.WAIT]

        state = self.extract_state(self.env)
        action_idx = self.select_action_index(state, explore=False)
        self.last_action_idx = action_idx
        return [ACTIONS[action_idx]]

    def save(self, checkpoint_path: str, map_paths: Optional[Sequence[str]] = None) -> None:
        target = Path(checkpoint_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "policy_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_steps": self.training_steps,
                "gamma": self.gamma,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "target_update_interval": self.target_update_interval,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay_steps": self.epsilon_decay_steps,
                "min_replay_size": self.min_replay_size,
                "replay_capacity": self.replay_buffer.capacity,
                "map_paths": list(map_paths or []),
                "input_dim": self.input_dim,
            },
            str(target),
        )

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        env: Optional[MazeEnvironment] = None,
        device: Optional[str] = None,
    ) -> "DQNAgent":
        checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
        agent = cls(
            env=env,
            gamma=float(checkpoint.get("gamma", 0.99)),
            learning_rate=float(checkpoint.get("learning_rate", 1e-3)),
            batch_size=int(checkpoint.get("batch_size", 128)),
            replay_capacity=int(checkpoint.get("replay_capacity", 120_000)),
            min_replay_size=int(checkpoint.get("min_replay_size", 2_000)),
            target_update_interval=int(checkpoint.get("target_update_interval", 500)),
            epsilon_start=float(checkpoint.get("epsilon_start", 1.0)),
            epsilon_end=float(checkpoint.get("epsilon_end", 0.05)),
            epsilon_decay_steps=int(checkpoint.get("epsilon_decay_steps", 75_000)),
            device=device,
        )

        agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        if "target_state_dict" in checkpoint:
            agent.target_net.load_state_dict(checkpoint["target_state_dict"])
        else:
            agent.target_net.load_state_dict(checkpoint["policy_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        agent.training_steps = int(checkpoint.get("training_steps", 0))
        return agent


def discover_map_paths(root: str) -> List[str]:
    base = Path(root)
    patterns = ["MAZE_*.png", "maze_*.png"]

    found: Set[Path] = set()
    for pattern in patterns:
        for path in base.glob(pattern):
            if path.is_file():
                found.add(path)

    return [str(path) for path in sorted(found)]
