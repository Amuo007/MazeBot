import heapq
import random


class HazardMap:
    def __init__(self, n_pits, n_teleports, open_cells, seed=None):
        self.open_cells = open_cells
        self.n_pits = n_pits
        self.n_teleports = n_teleports
        self._rng = random.Random(seed)
        self.pit_positions = set()
        self.teleport_positions = set()
        self._place()

    def _place(self):
        chosen = self._rng.sample(self.open_cells, self.n_pits + self.n_teleports)
        centered = []
        for cell in chosen:
            centered.append(self._center_in_corridor(cell))
        self.pit_positions = set(centered[:self.n_pits])
        self.teleport_positions = set(centered[self.n_pits:])

    def _center_in_corridor(self, cell):
        open_set = set(self.open_cells)
        r, c = cell

        left  = 0
        right = 0
        up    = 0
        down  = 0

        while (r, c - left - 1) in open_set:
            left += 1
        while (r, c + right + 1) in open_set:
            right += 1
        while (r - up - 1, c) in open_set:
            up += 1
        while (r + down + 1, c) in open_set:
            down += 1

        horiz_width = left + right
        vert_width  = up + down

        if horiz_width > vert_width:
            c = c - left + horiz_width // 2
        else:
            r = r - up + vert_width // 2

        return (r, c)

    def force_on_path(self, path_cells, n_pits, n_teleports):
        chosen = self._rng.sample(path_cells, n_pits + n_teleports)
        self.pit_positions = set(chosen[:n_pits])
        self.teleport_positions = set(chosen[n_pits:])

    @property
    def all_positions(self):
        return self.pit_positions | self.teleport_positions

    def is_hazard_at_or_near(self, cell, hazard_radius=0):
        r, c = cell
        for hr, hc in self.all_positions:
            if (r - hr) ** 2 + (c - hc) ** 2 <= hazard_radius ** 2:
                return True
        return False

    def get_triggering_hazard(self, cell, hazard_radius=0):
        r, c = cell
        for hazard in self.all_positions:
            hr, hc = hazard
            if (r - hr) ** 2 + (c - hc) ** 2 <= hazard_radius ** 2:
                return hazard
        return None


class MazeAgent:
    def __init__(self, start, goal, vision_range=20, stride=10, hazard_radius=0):
        self.start = start
        self.goal = goal
        self.position = start
        self.vision_range = vision_range
        self.stride = stride
        self.hazard_radius = hazard_radius
        self.hazard_detection = True
        self._rng = random.Random()

        self.path = []
        self.visited = [start]
        self.step_index = 0
        self.total_steps = 0
        self.replan_count = 0
        self.stuck = False

        self.known_hazards = set()
        self.last_explored = set()
        self.explored_fade = {}

        self.matrix = None
        self.hazard_map = None

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _astar(self, start, goal):
        matrix = self.matrix
        rows, cols = matrix.shape

        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g = {start: 0}
        explored = set()

        while open_set:
            _, cur = heapq.heappop(open_set)

            if cur in explored:
                continue
            explored.add(cur)

            if cur == goal:
                path = []
                node = cur
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append(start)
                path = path[::-1]

                self.last_explored = explored - set(path)
                for cell in self.last_explored:
                    self.explored_fade[cell] = 18

                return path

            r, c = cur
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                nb = (nr, nc)

                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if matrix[nr, nc] != 1:
                    continue
                if nb in explored:
                    continue

                ng = g[cur] + 1
                if ng < g.get(nb, float('inf')):
                    came_from[nb] = cur
                    g[nb] = ng
                    f = ng + self.heuristic(nb, goal)
                    heapq.heappush(open_set, (f, nb))

        self.last_explored = explored
        for cell in explored:
            self.explored_fade[cell] = 18
        return []

    def _replan(self, spotted_hazard_center, path_to_hazard=None):
        self.known_hazards.add(spotted_hazard_center)
        print(f"  👁 Hazard spotted at {spotted_hazard_center} — known: {len(self.known_hazards)}")

        rows, cols = self.matrix.shape

        protected = {self.goal}
        gr, gc = self.goal
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            protected.add((gr + dr, gc + dc))

        r0, c0 = spotted_hazard_center
        for dr in range(-self.hazard_radius, self.hazard_radius + 1):
            for dc in range(-self.hazard_radius, self.hazard_radius + 1):
                if dr * dr + dc * dc <= self.hazard_radius * self.hazard_radius:
                    nr, nc = r0 + dr, c0 + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if (nr, nc) not in protected and self.matrix[nr, nc] == 1:
                            self.matrix[nr, nc] = 0

        if path_to_hazard:
            for cell in path_to_hazard:
                if cell not in protected:
                    r, c = cell
                    if 0 <= r < rows and 0 <= c < cols:
                        if self.matrix[r, c] == 1:
                            self.matrix[r, c] = 0
            print(f"  🚧 Blocked {len(path_to_hazard)} corridor cells")

        pr, pc = self.position
        if self.position not in protected:
            self.matrix[pr, pc] = 0

        replan_start = self.position
        for cell in reversed(self.visited[:-1]):
            r, c = cell
            if self.matrix[r, c] == 1:
                replan_start = cell
                break

        new_path = self._astar(replan_start, self.goal)
        self.matrix[pr, pc] = 1

        if new_path:
            if replan_start != self.position:
                new_path = [self.position] + new_path[1:]
            self.path = new_path
            self.step_index = 0
            self.replan_count += 1
            print(f"  ↺ Replanned: {len(new_path)} cells from {self.position}")
        else:
            print("  ✗ DEAD END — no path exists")
            self.path = []
            self.step_index = 0
            self.stuck = True

    def _replan_from(self, new_start):
        print(f"  ⚡ TELEPORTED to {new_start} — replanning...")
        new_path = self._astar(new_start, self.goal)
        if new_path:
            self.path = new_path
            self.step_index = 0
            self.replan_count += 1
            print(f"  ↺ Replanned: {len(new_path)} cells from {new_start}")
        else:
            print("  ✗ DEAD END after teleport")
            self.path = []
            self.step_index = 0
            self.stuck = True

    def _check_hazard_at(self, pos):
        """Check if pos is on a hazard. Returns 'pit', 'teleport', or None."""
        trigger = self.hazard_map.get_triggering_hazard(pos, self.hazard_radius)
        if trigger is None:
            return None
        if trigger in self.hazard_map.pit_positions:
            return 'pit'
        if trigger in self.hazard_map.teleport_positions:
            return 'teleport'
        return None

    def find_path(self, matrix, hazard_map):
        self.matrix = matrix.copy()
        self.hazard_map = hazard_map
        self.path = self._astar(self.start, self.goal)
        self.step_index = 0

        if self.path:
            print(f"A*: initial path — {len(self.path)} cells (hazard-blind)")
            return True

        print("A*: No path found!")
        return False

    def take_steps(self, n=1):
        results = []

        expired = [c for c, t in self.explored_fade.items() if t <= 1]
        for c in expired:
            del self.explored_fade[c]
        for c in self.explored_fade:
            self.explored_fade[c] -= 1

        for _ in range(n):
            if self.stuck or len(self.path) == 0:
                break

            if self.step_index >= len(self.path) - 1:
                self.step_index = len(self.path) - 1
                self.position = self.path[self.step_index]
                self.visited.append(self.position)
                self.total_steps += 1
                results.append((self.position, None))
                break

            scan_end = min(self.step_index + self.vision_range, len(self.path))
            spotted_hazard = None
            spotted_idx = None

            if self.hazard_detection:
                for check_idx in range(self.step_index + 1, scan_end):
                    cell = self.path[check_idx]
                    trigger = self.hazard_map.get_triggering_hazard(cell, self.hazard_radius)
                    if trigger is not None and trigger not in self.known_hazards:
                        spotted_hazard = trigger
                        spotted_idx = check_idx
                        break

            if spotted_hazard is not None:
                stop_idx = max(self.step_index, spotted_idx - 1)
                self.step_index = stop_idx
                self.position = self.path[self.step_index]
                self.visited.append(self.position)
                self.total_steps += 1
                dead_end_corridor = self.path[self.step_index + 1: spotted_idx]
                self._replan(spotted_hazard, path_to_hazard=dead_end_corridor)
                results.append((self.position, 'spotted'))
                break

            next_idx = min(self.step_index + self.stride, len(self.path) - 1)

            blocked = False
            if self.hazard_detection:
                for idx in range(self.step_index + 1, next_idx + 1):
                    cell = self.path[idx]
                    trigger = self.hazard_map.get_triggering_hazard(cell, self.hazard_radius)
                    if trigger is not None and trigger not in self.known_hazards:
                        self.step_index = idx - 1
                        self.position = self.path[self.step_index]
                        self.visited.append(self.position)
                        self.total_steps += 1
                        dead_end_corridor = self.path[self.step_index + 1: idx]
                        self._replan(trigger, path_to_hazard=dead_end_corridor)
                        results.append((self.position, 'spotted'))
                        blocked = True
                        break

            if blocked:
                break

            self.step_index = next_idx
            self.position = self.path[self.step_index]
            self.visited.append(self.position)
            self.total_steps += 1

            # Check if stepped ON a hazard
            hazard_type = self._check_hazard_at(self.position)

            if hazard_type == 'pit':
                print(f"  💀 FELL INTO PIT at {self.position}")
                results.append((self.position, 'pit'))
                self.stuck = True
                break
            elif hazard_type == 'teleport':
                print(f"  ⚡ HIT TELEPORT at {self.position}")
                new_pos = self._rng.choice(self.hazard_map.open_cells)
                self.position = new_pos
                self.visited.append(self.position)
                self._replan_from(new_pos)
                results.append((new_pos, 'teleport'))
                break
            else:
                results.append((self.position, None))

        return results

    def is_done(self):
        return (
            len(self.path) > 0
            and self.step_index >= len(self.path) - 1
            and self.position == self.goal
        )

    def get_position(self):
        return self.position

    def get_remaining_path(self):
        return self.path[self.step_index:]