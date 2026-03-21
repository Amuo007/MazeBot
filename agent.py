import heapq

EMPTY = 0
FIRE = 1
CONFUSION = 2
TP_PURPLE = 3
TP_RED = 4
TP_GREEN = 5
START = 6
GOAL = 7
UNKNOWN = 99


class MazeAgent:
    def __init__(
        self,
        start,
        goal,
        vertical_walls,
        horizontal_walls,
        obj_matrix,
        teleport_pairs,
        fire_phase_sets,
    ):
        self.start = start
        self.goal = goal
        self.position = start

        self.vertical_walls = vertical_walls
        self.horizontal_walls = horizontal_walls
        self.obj_matrix = obj_matrix
        self.teleport_pairs = teleport_pairs
        self.fire_phase_sets = fire_phase_sets

        self.rows, self.cols = obj_matrix.shape

        self.path = []
        self.path_index = 0
        self.visited = [start]

        self.total_steps = 0
        self.replans = 0
        self.dead = False
        self.done = False

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def in_bounds(self, cell):
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def get_active_fire_cells(self):
        phase = (self.total_steps // 5) % 4
        return self.fire_phase_sets[phase]

    def is_blocked_cell(self, cell):
        return False

    def can_move(self, a, b):
        ar, ac = a
        br, bc = b

        if not self.in_bounds(b):
            return False

        if br == ar - 1 and bc == ac:
            return self.horizontal_walls[ar, ac] == 0

        if br == ar + 1 and bc == ac:
            return self.horizontal_walls[ar + 1, ac] == 0

        if br == ar and bc == ac - 1:
            return self.vertical_walls[ar, ac] == 0

        if br == ar and bc == ac + 1:
            return self.vertical_walls[ar, ac + 1] == 0

        return False

    def neighbors(self, cell):
        r, c = cell
        out = []
        for nb in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if self.can_move(cell, nb):
                out.append(nb)
        return out

    def astar(self, start, goal):
        open_heap = []
        heapq.heappush(open_heap, (0, start))

        came_from = {}
        g_score = {start: 0}
        closed = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)

            if current in closed:
                continue
            closed.add(current)

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for nb in self.neighbors(current):
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(nb, float("inf")):
                    came_from[nb] = current
                    g_score[nb] = tentative_g
                    f = tentative_g + self.heuristic(nb, goal)
                    heapq.heappush(open_heap, (f, nb))

        return []

    def plan_from(self, pos):
        self.path = self.astar(pos, self.goal)
        self.path_index = 0
        return len(self.path) > 0

    def find_initial_path(self):
        return self.plan_from(self.start)

    def get_remaining_path(self):
        if not self.path:
            return []
        return self.path[self.path_index:]

    def step(self):
        if self.dead:
            return "dead"

        if self.done:
            return "goal"

        if not self.path:
            self.dead = True
            return "dead"

        if self.position == self.goal:
            self.done = True
            return "goal"

        if self.path_index >= len(self.path) - 1:
            self.position = self.path[-1]
            if self.position != self.visited[-1]:
                self.visited.append(self.position)
            if self.position == self.goal:
                self.done = True
                return "goal"
            return "move"

        self.path_index += 1
        self.position = self.path[self.path_index]
        self.visited.append(self.position)
        self.total_steps += 1

        if self.position in self.get_active_fire_cells():
            self.dead = True
            return "dead"

        r, c = self.position
        tile = self.obj_matrix[r, c]

        if self.position == self.goal or tile == GOAL:
            self.done = True
            return "goal"

        if self.position in self.teleport_pairs:
            self.position = self.teleport_pairs[self.position]
            self.visited.append(self.position)
            self.replans += 1

            if self.position in self.get_active_fire_cells():
                self.dead = True
                return "dead"

            if self.position == self.goal:
                self.done = True
                return "goal"

            ok = self.plan_from(self.position)
            if not ok:
                self.dead = True
                return "dead"

            return "teleport"

        return "move"