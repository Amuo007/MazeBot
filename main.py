from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from agent import MazeAgent
from dqn_agent import DQNAgent
from environment import MazeEnvironment
from visualizer import animate_episode


DEFAULT_IMAGE = "maze_5_edited.png"
DEFAULT_CHECKPOINT = "checkpoints/dqn_maze.pt"


def run_astar(image_path: str, max_turns: int, frame_ms: int) -> None:
    env = MazeEnvironment(image_path=image_path, maze_size=64)
    agent = MazeAgent(
        start=env.start,
        goal=env.goal,
        vertical_walls=env.vertical_walls,
        horizontal_walls=env.horizontal_walls,
        obj_matrix=env.obj_matrix,
        teleport_pairs=env.teleport_pairs,
    )

    env.reset()
    agent.reset_episode()

    print(f"[astar] map={image_path}")
    print(f"[astar] start={env.start} goal={env.goal}")
    animate_episode(env, agent, max_turns=max_turns, frame_ms=frame_ms)
    print("[astar] episode stats:", env.get_episode_stats())


def parse_map_arg(raw: str) -> List[str]:
    maps = [entry.strip() for entry in raw.split(",") if entry.strip()]
    unique = []
    seen = set()
    for item in maps:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique


def resolve_maps(args: argparse.Namespace, project_root: Path) -> List[str]:
    if args.maps:
        map_paths = parse_map_arg(args.maps)
    elif args.image:
        map_paths = [args.image]
    else:
        map_paths = [str(project_root / DEFAULT_IMAGE)]

    existing = [path for path in map_paths if Path(path).exists()]
    if not existing:
        raise FileNotFoundError(
            f"No valid map PNG files were found. Default expected map: {DEFAULT_IMAGE}"
        )
    return existing


def run_dqn_train(args: argparse.Namespace, project_root: Path) -> None:
    map_paths = resolve_maps(args, project_root)
    print(f"[train] maps={len(map_paths)}")
    for path in map_paths:
        print(f"  - {path}")

    agent = DQNAgent()
    history = agent.train(
        map_paths=map_paths,
        episodes=args.episodes,
        max_turns=args.max_turns,
        log_every=max(1, args.log_every),
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        checkpoint_every=25,
    )
    agent.save(args.checkpoint, map_paths=map_paths)

    final_success = sum(history["success"][-50:]) / max(1, len(history["success"][-50:]))
    print(f"[train] saved checkpoint -> {args.checkpoint}")
    print(f"[train] final_success_last_50={final_success:.3f}")

    if args.eval_episodes > 0:
        metrics = agent.evaluate(
            map_paths=map_paths,
            episodes_per_map=args.eval_episodes,
            max_turns=args.max_turns,
        )
        print("[train] post-train eval")
        for map_path, values in metrics.items():
            print(
                f"  {map_path}: success={values['success_rate']:.2f} "
                f"avg_steps={values['avg_steps']:.1f} avg_deaths={values['avg_deaths']:.2f}"
            )


def run_dqn_play(args: argparse.Namespace) -> None:
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    image_path = args.image or DEFAULT_IMAGE
    env = MazeEnvironment(image_path=image_path, maze_size=64)
    env.reset()

    agent = DQNAgent.load_from_checkpoint(args.checkpoint, env=env)
    agent.reset_episode()

    print(f"[play] map={image_path}")
    print(f"[play] checkpoint={args.checkpoint}")
    animate_episode(env, agent, max_turns=args.max_turns, frame_ms=args.frame_ms)
    print("[play] episode stats:", env.get_episode_stats())


def run_dqn_eval(args: argparse.Namespace, project_root: Path) -> None:
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    map_paths = resolve_maps(args, project_root)
    agent = DQNAgent.load_from_checkpoint(args.checkpoint)

    metrics = agent.evaluate(
        map_paths=map_paths,
        episodes_per_map=max(1, args.eval_episodes),
        max_turns=args.max_turns,
    )

    print("[eval] results")
    for map_path, values in metrics.items():
        print(
            f"  {map_path}: success={values['success_rate']:.2f} "
            f"avg_steps={values['avg_steps']:.1f} avg_deaths={values['avg_deaths']:.2f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MazeBot A* + DQN runner")
    parser.add_argument("--mode", choices=["astar", "train", "play", "eval"], default="train")
    parser.add_argument("--image", default="", help="Single map image path for astar/play. For train/eval, acts as a single-map fallback if --maps is not provided.")
    parser.add_argument(
        "--maps",
        default="",
        help="Comma-separated map image paths for train/eval. If omitted and --image is not set, defaults to maze_5_edited.png only.",
    )
    parser.add_argument("--episodes", type=int, default=800, help="Training episodes")
    parser.add_argument("--eval-episodes", type=int, default=3, help="Evaluation episodes per map")
    parser.add_argument("--max-turns", type=int, default=1500)
    parser.add_argument("--frame-ms", type=int, default=90)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=25)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent

    if args.mode == "astar":
        image_path = args.image or DEFAULT_IMAGE
        run_astar(image_path, args.max_turns, args.frame_ms)
        return

    if args.mode == "train":
        run_dqn_train(args, project_root)
        return

    if args.mode == "play":
        run_dqn_play(args)
        return

    if args.mode == "eval":
        run_dqn_eval(args, project_root)
        return

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()