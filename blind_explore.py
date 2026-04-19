from __future__ import annotations

import heapq
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from environment import (
    MazeEnvironment,
    Action,
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

# ============================================================
# CONFIG
# ============================================================
IMAGE_PATH = "maze_9.png"
MAZE_SIZE = 64
MAX_PHYSICAL_STEPS = 20000

# animation
FRAME_MS = 15
SNAPSHOTS_PER_FRAME = 3
PRINT_EVERY = 250

# ============================================================
# DISPLAY COLORS
# ============================================================
DISPLAY_COLORS = {
    EMPTY:      np.array([1.00, 1.00, 1.00]),
    FIRE:       np.array([255, 145, 76]) / 255.0,
    CONFUSION:  np.array([255, 222, 89]) / 255.0,
    TP_PURPLE:  np.array([140, 82, 255]) / 255.0,
    TP_RED:     np.array([255, 49, 50]) / 255.0,
    TP_GREEN:   np.array([1, 191, 99]) / 255.0,
    START:      np.array([15, 192, 223]) / 255.0,
    GOAL:       np.array([0, 74, 173]) / 255.0,
    UNKNOWN:    np.array([0.68, 0.68, 0.68]),
}
COL_VISITED  = np.array([0.60, 0.82, 1.00])
COL_TARGET   = np.array([1.00, 0.70, 0.10])
COL_AGENT    = np.array([1.00, 0.10, 0.10])
COL_DANGER   = np.array([0.45, 0.10, 0.10])
COL_PATH     = np.array([1.00, 0.55, 0.15])


# ============================================================
# SMALL UTILS
# ============================================================
def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def adjacent(a: Cell, b: Cell) -> bool:
    return manhattan(a, b) == 1


def delta_to_action(a: Cell, b: Cell) -> Action:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    if dr == -1 and dc == 0: return Action.MOVE_UP
    if dr ==  1 and dc == 0: return Action.MOVE_DOWN
    if dr ==  0 and dc == -1: return Action.MOVE_LEFT
    if dr ==  0 and dc ==  1: return Action.MOVE_RIGHT
    return Action.WAIT


def invert_action(a: Action) -> Action:
    if a == Action.MOVE_UP:    return Action.MOVE_DOWN
    if a == Action.MOVE_DOWN:  return Action.MOVE_UP
    if a == Action.MOVE_LEFT:  return Action.MOVE_RIGHT
    if a == Action.MOVE_RIGHT: return Action.MOVE_LEFT
    return Action.WAIT


# ============================================================
# BLIND KNOWLEDGE
#   Only holds what the agent has personally observed.
#   No ground-truth reads from env.vertical_walls / obj_matrix
#   ever happen here.
# ============================================================
class BlindKnowledge:
    def __init__(self, maze_size: int, start: Cell, goal: Cell):
        self.n = maze_size
        self.start = start
        self.goal = goal  # goal coord is assumed known (like in a real search problem)

        self.visited: Set[Cell] = {start}
        self.tile_of: Dict[Cell, int] = {start: START}

        # edge knowledge (stored bidirectionally)
        self.open_edges: Set[Tuple[Cell, Cell]] = set()
        self.blocked_edges: Set[Tuple[Cell, Cell]] = set()

        # teleport pairs we've actually been bounced through
        self.teleport_pairs: Dict[Cell, Cell] = {}

        # cells we stepped into and died in -> treat as lethal in planning
        self.dangerous: Set[Cell] = set()
        # probes we've given up on (unreachable right now)
        self.abandoned: Set[Cell] = set()

        # confusion cells seen (visual only)
        self.confusion_cells: Set[Cell] = set()

    # ---- geometry ----
    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.n and 0 <= c < self.n

    def candidate_neighbors(self, cell: Cell) -> List[Cell]:
        r, c = cell
        return [nb for nb in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)] if self.in_bounds(nb)]

    def open_neighbors(self, cell: Cell) -> List[Cell]:
        return [nb for nb in self.candidate_neighbors(cell) if (cell, nb) in self.open_edges]

    def planning_neighbors(self, cell: Cell) -> List[Cell]:
        """
        Graph used by A* over what the agent knows so far.
        Walking from `cell` into an adjacent known-open neighbour `nb` lands
        us at `nb` — unless `nb` is a known teleporter, in which case the
        env bounces us to `teleport_pairs[nb]`. So the true physical
        destination when we step toward `nb` is its teleport-resolved
        landing.
        """
        out = []
        for nb in self.open_neighbors(cell):
            landing = self.teleport_pairs.get(nb, nb)
            out.append(landing)
        return out

    def find_walk_step(self, cur: Cell, target_landing: Cell) -> Optional[Cell]:
        """Which adjacent walkable cell do I step onto to end up at target_landing
        (possibly via teleport)?"""
        for nb in self.open_neighbors(cur):
            landing = self.teleport_pairs.get(nb, nb)
            if landing == target_landing:
                return nb
        return None

    # ---- edge updates ----
    def mark_open(self, a: Cell, b: Cell) -> None:
        self.open_edges.add((a, b))
        self.open_edges.add((b, a))

    def mark_blocked(self, a: Cell, b: Cell) -> None:
        self.blocked_edges.add((a, b))
        self.blocked_edges.add((b, a))

    # ---- frontier ----
    def is_probe_candidate(self, v: Cell, nb: Cell) -> bool:
        if nb in self.visited:
            return False
        if (v, nb) in self.blocked_edges:
            return False
        if nb in self.dangerous:
            return False
        if nb in self.abandoned:
            return False
        return True

    def reachable_from(self, start: Cell) -> Set[Cell]:
        """BFS over known passable graph (open edges + known teleports)."""
        seen = {start}
        stack = [start]
        while stack:
            cur = stack.pop()
            for nb in self.planning_neighbors(cur):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return seen

    def pick_frontier_probe(self, current: Cell) -> Optional[Tuple[Cell, Cell]]:
        """
        Greedy best-first frontier pick.

        Score each candidate probe (v, nb) by
            cost-so-far-to-reach-v  +  heuristic-from-nb-to-goal
        (both Manhattan). This makes the agent behave like A* running on
        the frontier — it still aims toward the goal, but it won't abandon
        a nearby unexplored edge to chase a distant one whose h is only
        marginally better.
        """
        reachable = self.reachable_from(current)
        best = None
        best_key = None
        for v in self.visited:
            if v not in reachable:
                continue
            for nb in self.candidate_neighbors(v):
                if not self.is_probe_candidate(v, nb):
                    continue
                f = manhattan(v, current) + manhattan(nb, self.goal)
                key = (f, manhattan(nb, self.goal))
                if best_key is None or key < best_key:
                    best_key = key
                    best = (v, nb)
        return best


# ============================================================
# A* OVER KNOWN GRAPH ONLY
# ============================================================
def plan_known_route(knowledge: BlindKnowledge, start: Cell, target: Cell) -> Optional[List[Cell]]:
    if start == target:
        return [start]

    open_heap: List[Tuple[int, int, int, Cell]] = []
    heapq.heappush(open_heap, (manhattan(start, target), 0, 0, start))

    came_from: Dict[Cell, Cell] = {}
    g_score: Dict[Cell, int] = {start: 0}
    closed: Set[Cell] = set()
    counter = 0

    while open_heap:
        _, g, _, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)

        if cur == target:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path

        for nb in knowledge.planning_neighbors(cur):
            if nb in knowledge.dangerous and nb != target:
                continue
            tentative = g + 1
            if tentative < g_score.get(nb, 10**9):
                g_score[nb] = tentative
                came_from[nb] = cur
                counter += 1
                heapq.heappush(open_heap, (tentative + manhattan(nb, target), tentative, counter, nb))

    return None


def shortest_path_in_discovered(knowledge: BlindKnowledge) -> List[Cell]:
    """BFS over discovered passable graph (used for final summary path)."""
    q = deque([knowledge.start])
    came_from: Dict[Cell, Cell] = {knowledge.start: knowledge.start}
    while q:
        cur = q.popleft()
        if cur == knowledge.goal:
            path = [cur]
            while came_from[cur] != cur:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path
        for nb in knowledge.planning_neighbors(cur):
            if nb in came_from:
                continue
            came_from[nb] = cur
            q.append(nb)
    return []


# ============================================================
# BLIND EXPLORER
# ============================================================
class BlindExplorer:
    def __init__(self, env: MazeEnvironment):
        self.env = env
        self.knowledge = BlindKnowledge(env.maze_size, env.start, env.goal)
        self.current: Cell = env.start
        self.physical_steps = 0
        self.deaths = 0
        self.wall_bumps = 0
        self.teleports_taken = 0
        self.confusions_entered = 0
        self.history: List[dict] = []
        self._snapshot(event="start", target=None)

    # ---------------- snapshot for animation ----------------
    def _snapshot(self, event: str, target: Optional[Cell]) -> None:
        self.history.append({
            "current": self.current,
            "visited": set(self.knowledge.visited),
            "blocked_edges": set(self.knowledge.blocked_edges),
            "dangerous": set(self.knowledge.dangerous),
            "confusion": set(self.knowledge.confusion_cells),
            "teleports": dict(self.knowledge.teleport_pairs),
            "active_fire": set(self.env.get_active_fire_cells()),
            "fire_phase": self.env.get_fire_phase(),
            "target": target,
            "event": event,
            "steps": self.physical_steps,
            "deaths": self.deaths,
            "wall_bumps": self.wall_bumps,
            "teleports_taken": self.teleports_taken,
        })

    # ---------------- one physical action ----------------
    def _step_physical(self, next_cell: Cell, target: Optional[Cell]) -> str:
        """
        Execute a single physical move from self.current toward an adjacent next_cell.
        Handles active confusion inversion. Updates knowledge based on the TurnResult.
        Returns one of: 'moved', 'wall', 'dead', 'diverged'.
        """
        if not adjacent(self.current, next_cell):
            return "diverged"

        planned = delta_to_action(self.current, next_cell)
        # If the env is about to apply confusion inversion to this turn,
        # pre-invert so the effective action is what we actually want.
        if self.env.confused_turns_remaining > 0:
            action_to_send = invert_action(planned)
        else:
            action_to_send = planned

        before = self.current
        result = self.env.step([action_to_send])
        self.physical_steps += 1

        # --- death in fire ---
        if result.is_dead:
            # We died stepping into next_cell (or the teleport target of it);
            # mark the place we aimed at as dangerous.
            self.knowledge.dangerous.add(next_cell)
            self.current = self.env.start
            self.deaths += 1
            self._snapshot(event="death", target=target)
            return "dead"

        # --- bumped into a wall, didn't move ---
        if result.wall_hits > 0:
            self.knowledge.mark_blocked(before, next_cell)
            self.wall_bumps += 1
            self._snapshot(event="wall", target=target)
            return "wall"

        # --- successful move (possibly with teleport) ---
        landing = result.current_position
        self.knowledge.mark_open(before, next_cell)
        self.knowledge.visited.add(next_cell)
        self.knowledge.tile_of[next_cell] = int(self.env.obj_matrix[next_cell])

        if result.teleported:
            # next_cell was the teleporter we stepped on; landing is the paired tile.
            self.knowledge.teleport_pairs[next_cell] = landing
            self.knowledge.teleport_pairs[landing] = next_cell
            self.knowledge.visited.add(landing)
            self.knowledge.tile_of[landing] = int(self.env.obj_matrix[landing])
            self.teleports_taken += 1

        if int(self.env.obj_matrix[next_cell]) == CONFUSION:
            self.knowledge.confusion_cells.add(next_cell)
            self.confusions_entered += 1

        self.current = landing
        self._snapshot(event="moved", target=target)
        return "moved"

    # ---------------- main exploration loop ----------------
    def run(self, max_steps: int = MAX_PHYSICAL_STEPS) -> bool:
        while self.physical_steps < max_steps:
            if self.current == self.knowledge.goal:
                return True

            probe = self.knowledge.pick_frontier_probe(self.current)
            if probe is None:
                # no frontier reachable — we are stuck on an island
                return False

            probe_from, probe_to = probe

            route = plan_known_route(self.knowledge, self.current, probe_from)
            if route is None:
                # shouldn't happen — reachable filter guarantees it, but be safe
                self.knowledge.abandoned.add(probe_to)
                continue

            # walk the known route
            aborted = False
            i = 0
            while i < len(route) and self.current == route[i]:
                i += 1

            while i < len(route):
                nxt = route[i]
                if self.current == nxt:
                    i += 1
                    continue

                # The route is expressed in terms of *landing* cells.
                # Figure out which adjacent walkable cell to step onto so we
                # end up at `nxt` — this may be `nxt` itself, or a teleporter
                # whose pair is `nxt`.
                step_cell = self.knowledge.find_walk_step(self.current, nxt)
                if step_cell is None:
                    # plan desynced with reality; bail and let outer loop replan
                    self.knowledge.abandoned.add(probe_to)
                    aborted = True
                    break

                event = self._step_physical(step_cell, target=probe_to)
                if event != "moved":
                    aborted = True
                    break
                if self.current == self.knowledge.goal:
                    return True
                if self.physical_steps >= max_steps:
                    return False

                while i < len(route) and self.current == route[i]:
                    i += 1

            if aborted:
                continue

            # now physically probe the unknown edge (probe_from -> probe_to)
            if self.current == probe_from:
                event = self._step_physical(probe_to, target=probe_to)

                if self.physical_steps % PRINT_EVERY == 0:
                    print(
                        f"[EXPLORE] steps={self.physical_steps:5d} "
                        f"visited={len(self.knowledge.visited):4d} "
                        f"walls={len(self.knowledge.blocked_edges)//2:4d} "
                        f"deaths={self.deaths:3d} "
                        f"tp={self.teleports_taken:3d} "
                        f"target={probe_to} event={event}"
                    )

                if self.current == self.knowledge.goal:
                    return True

        return self.current == self.knowledge.goal


# ============================================================
# ANIMATION
# ============================================================
def build_display(env: MazeEnvironment, snap: dict) -> np.ndarray:
    n = env.maze_size
    disp = np.ones((n, n, 3), dtype=float) * DISPLAY_COLORS[UNKNOWN]

    # visited cells reveal their real tile, then tinted blue
    for (r, c) in snap["visited"]:
        tile = int(env.obj_matrix[r, c])
        base = DISPLAY_COLORS.get(tile, DISPLAY_COLORS[EMPTY])
        if tile == FIRE:
            base = DISPLAY_COLORS[EMPTY]
        disp[r, c] = base * 0.45 + COL_VISITED * 0.55

    # dangerous cells (fire deaths) darken
    for (r, c) in snap["dangerous"]:
        disp[r, c] = COL_DANGER

    # active fire (viewer sees the current phase even if agent doesn't)
    for (r, c) in snap["active_fire"]:
        disp[r, c] = DISPLAY_COLORS[FIRE]

    # always show start + goal
    sr, sc = env.start
    gr, gc = env.goal
    disp[sr, sc] = DISPLAY_COLORS[START]
    disp[gr, gc] = DISPLAY_COLORS[GOAL]

    # current frontier target highlight
    tgt = snap.get("target")
    if tgt is not None:
        disp[tgt[0], tgt[1]] = disp[tgt[0], tgt[1]] * 0.25 + COL_TARGET * 0.75

    # agent on top
    cr, cc = snap["current"]
    disp[cr, cc] = COL_AGENT

    return disp


def discovered_walls_to_xy(edges: Set[Tuple[Cell, Cell]]):
    xs: List = []
    ys: List = []
    drawn = set()
    for a, b in edges:
        key = tuple(sorted([a, b]))
        if key in drawn:
            continue
        drawn.add(key)
        ar, ac = a
        br, bc = b
        if ar == br:  # horizontal neighbors -> draw vertical wall between them
            col = max(ac, bc)
            xs += [col, col, None]
            ys += [ar, ar + 1, None]
        else:         # vertical neighbors -> draw horizontal wall between them
            row = max(ar, br)
            xs += [ac, ac + 1, None]
            ys += [row, row, None]
    return xs, ys


def animate(env: MazeEnvironment, explorer: BlindExplorer, final_path: List[Cell]) -> None:
    history = explorer.history
    n = env.maze_size

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.subplots_adjust(right=0.78)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    im = ax.imshow(
        build_display(env, history[0]),
        extent=(0, n, n, 0),
        interpolation="nearest",
    )

    # outer maze boundary
    ax.plot([0, n, n, 0, 0], [0, 0, n, n, 0], color="black", linewidth=1.2)
    # wall overlay — updated each frame
    wall_lines, = ax.plot([], [], color="black", linewidth=0.9)
    # path overlay (drawn at the end)
    path_line, = ax.plot([], [], color="orange", linewidth=2.2)

    title = ax.set_title("Blind exploration", fontsize=10)
    stats_box = ax.text(
        1.02, 0.98, "",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=10, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", alpha=0.95),
    )

    state = {"idx": 0, "done": False}
    ani_holder = {"ani": None}

    def draw_final_path_if_any():
        if not final_path:
            return
        xs_p = [c + 0.5 for (_, c) in final_path]
        ys_p = [r + 0.5 for (r, _) in final_path]
        path_line.set_data(xs_p, ys_p)

    def update(_frame):
        if state["done"]:
            return [im, title, stats_box, wall_lines, path_line]

        state["idx"] = min(len(history) - 1, state["idx"] + SNAPSHOTS_PER_FRAME)
        snap = history[state["idx"]]

        im.set_data(build_display(env, snap))
        xs, ys = discovered_walls_to_xy(snap["blocked_edges"])
        wall_lines.set_data(xs, ys)

        title.set_text(
            f"Blind GBFS exploration | step {snap['steps']} | "
            f"visited {len(snap['visited'])} | "
            f"deaths {snap['deaths']} | "
            f"event {snap['event']}"
        )
        stats_box.set_text("\n".join([
            f"steps:        {snap['steps']}",
            f"visited:      {len(snap['visited'])}",
            f"walls found:  {len(snap['blocked_edges']) // 2}",
            f"deaths:       {snap['deaths']}",
            f"wall bumps:   {snap['wall_bumps']}",
            f"teleports:    {snap['teleports_taken']}",
            f"fire_phase:   {snap['fire_phase']}",
            f"event:        {snap['event']}",
            f"target:       {snap.get('target')}",
        ]))

        if state["idx"] >= len(history) - 1:
            state["done"] = True
            draw_final_path_if_any()
            if ani_holder["ani"] is not None:
                ani_holder["ani"].event_source.stop()

        return [im, title, stats_box, wall_lines, path_line]

    total_frames = max(1, len(history) // max(1, SNAPSHOTS_PER_FRAME)) + 20
    ani_holder["ani"] = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=FRAME_MS, blit=False, repeat=False,
    )
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================
def main():
    env = MazeEnvironment(image_path=IMAGE_PATH, maze_size=MAZE_SIZE)
    print(f"Start: {env.start}")
    print(f"Goal : {env.goal}\n")

    print("Running BLIND greedy best-first exploration...")
    print("  (agent only knows start/goal coords; discovers walls by bumping,\n"
          "   tiles by stepping, teleport pairs by being bounced, fire by dying)\n")

    explorer = BlindExplorer(env)
    reached = explorer.run(max_steps=MAX_PHYSICAL_STEPS)

    print()
    if reached:
        print("GOAL REACHED")
    else:
        print("GOAL NOT REACHED (step budget exhausted or stuck)")

    print(f"Physical steps       : {explorer.physical_steps}")
    print(f"Cells discovered     : {len(explorer.knowledge.visited)}")
    print(f"Walls discovered     : {len(explorer.knowledge.blocked_edges) // 2}")
    print(f"Deaths in fire       : {explorer.deaths}")
    print(f"Wall bumps           : {explorer.wall_bumps}")
    print(f"Teleports taken      : {explorer.teleports_taken}")
    print(f"Confusion tiles hit  : {explorer.confusions_entered}")

    final_path = shortest_path_in_discovered(explorer.knowledge) if reached else []
    if final_path:
        print(f"Shortest path in discovered graph : {len(final_path) - 1} steps")

    animate(env, explorer, final_path)


if __name__ == "__main__":
    main()
