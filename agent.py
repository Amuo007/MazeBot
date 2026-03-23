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

        self.confused_turns_remaining = 0
        self.confusion_count = 0
        self.confusion_locations = []
        self.death_count = 0
        self.known_pits = set()
        self.pending_respawn = False

        self.total_steps = 0
        self.replans = 0
        self.dead = False
        self.done = False
        self.failed = False

    @property
    def is_confused(self):
        return self.confused_turns_remaining > 0

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def in_bounds(self, cell):
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def get_active_fire_cells(self):
        phase = (self.total_steps // 5) % 4
        return self.fire_phase_sets[phase]

    def is_blocked_cell(self, cell):
        if cell == self.start:
            return False
        return cell in self.known_pits

    def _activate_confusion(self, cell):
        self.confusion_count += 1
        self.confusion_locations.append(cell)
        self.confused_turns_remaining = 2

    def _trigger_death(self, cell):
        self.death_count += 1
        self.known_pits.add(cell)
        self.dead = True
        self.pending_respawn = True

    def _delta(self, a, b):
        return (b[0] - a[0], b[1] - a[1])

    def _apply_delta(self, cell, delta):
        return (cell[0] + delta[0], cell[1] + delta[1])

    def can_move(self, a, b):
        ar, ac = a
        br, bc = b

        if not self.in_bounds(b):
            return False

        if self.is_blocked_cell(b):
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
        if self.failed:
            return "stuck"

        if self.pending_respawn:
            self.position = self.start
            self.dead = False
            self.pending_respawn = False
            self.confused_turns_remaining = 0

            if self.position != self.visited[-1]:
                self.visited.append(self.position)

            ok = self.plan_from(self.position)
            if not ok:
                self.failed = True
                return "stuck"

            return "respawn"

        if self.dead:
            return "dead"

        if self.done:
            return "goal"

        if not self.path:
            self.failed = True
            return "stuck"

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

        intended_next = self.path[self.path_index + 1]
        event = "move"

        planned_delta = self._delta(self.position, intended_next)
        command_delta = planned_delta

        if self.is_confused:
            # Confusion flips controls, so we issue the opposite command
            # to track the original plan.
            command_delta = (-planned_delta[0], -planned_delta[1])
            actual_delta = (-command_delta[0], -command_delta[1])
            self.confused_turns_remaining = max(0, self.confused_turns_remaining - 1)
        else:
            actual_delta = command_delta

        actual_next = self._apply_delta(self.position, actual_delta)
        if not self.can_move(self.position, actual_next):
            self.failed = True
            return "stuck"

        self.path_index += 1
        self.position = actual_next

        if self.position != self.visited[-1]:
            self.visited.append(self.position)
        self.total_steps += 1

        if self.position in self.get_active_fire_cells():
            self._trigger_death(self.position)
            return "dead"

        r, c = self.position
        tile = self.obj_matrix[r, c]

        if self.position == self.goal or tile == GOAL:
            self.done = True
            return "goal"

        if tile == CONFUSION:
            self._activate_confusion(self.position)

        if self.position in self.teleport_pairs:
            self.position = self.teleport_pairs[self.position]
            self.visited.append(self.position)
            self.replans += 1
            event = "teleport"

            if self.position in self.get_active_fire_cells():
                self._trigger_death(self.position)
                return "dead"

            tile = self.obj_matrix[self.position[0], self.position[1]]

            if self.position == self.goal:
                self.done = True
                return "goal"

            if tile == CONFUSION:
                self._activate_confusion(self.position)

            ok = self.plan_from(self.position)
            if not ok:
                self.failed = True
                return "stuck"

            return event

        return event