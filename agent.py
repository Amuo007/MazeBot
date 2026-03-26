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
        self.visited_mask = None

        self.confused_turns_remaining = 0
        self.confusion_count = 0
        self.confusion_locations = []
        self.Episodes = 0
        self.known_pits = set()
        self.known_pit_phases = {}
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
        phase = (self.total_steps // 5) % len(self.fire_phase_sets)
        return self.fire_phase_sets[phase]

    def get_fire_cells_for_step(self, step_count):
        phase = (step_count // 5) % len(self.fire_phase_sets)
        return self.fire_phase_sets[phase]

    def is_fire_at_step(self, cell, step_count):
        return cell in self.get_fire_cells_for_step(step_count)

    def is_blocked_cell(self, cell):
        return False

    def is_phase_blocked_cell(self, cell, step_count):
        if cell == self.start:
            return False

        phase = (step_count // 5) % len(self.fire_phase_sets)
        phases = self.known_pit_phases.get(cell)
        return phases is not None and phase in phases

    def _activate_confusion(self, cell):
        self.confusion_count += 1
        self.confusion_locations.append(cell)
        self.confused_turns_remaining = 2

    def _trigger_death(self, cell):
        self.Episodes += 1
        self.known_pits.add(cell)
        phase = (self.total_steps // 5) % len(self.fire_phase_sets)
        self.known_pit_phases.setdefault(cell, set()).add(phase)
        self.dead = True
        self.pending_respawn = True

    def _respawn_now(self):
        self.position = self.start
        self.dead = False
        self.pending_respawn = False
        self.confused_turns_remaining = 0

        # If start is unsafe in this phase, keep a deferred respawn state.
        if self.position in self.get_active_fire_cells():
            self.dead = True
            self.pending_respawn = True
            return "dead"

        ok = self.plan_from(self.position)
        if ok:
            return "respawn"

        # No route at this phase; stay alive and try again on future turns.
        return "wait"

    def _delta(self, a, b):
        return (b[0] - a[0], b[1] - a[1])

    def _apply_delta(self, cell, delta):
        return (cell[0] + delta[0], cell[1] + delta[1])

    def can_move(self, a, b):
        ar, ac = a
        br, bc = b

        if a == b:
            return True

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

    def time_neighbors(self, cell):
        # Include waiting in place so the agent can let rotating hazards pass.
        return [cell] + self.neighbors(cell)

    def astar(self, start, goal, start_step):
        period = 5 * len(self.fire_phase_sets)
        open_heap = []
        start_state = (start, start_step % period)
        heapq.heappush(open_heap, (0, 0, start_state))

        came_from = {}
        g_score = {start_state: 0}
        closed = set()

        while open_heap:
            _, _, state = heapq.heappop(open_heap)
            current, t_mod = state

            if state in closed:
                continue
            closed.add(state)

            if current == goal:
                path = [current]
                cursor = state
                while cursor in came_from:
                    cursor = came_from[cursor]
                    path.append(cursor[0])
                path.reverse()
                return path

            current_g = g_score[state]
            for nb in self.time_neighbors(current):
                next_step = start_step + current_g + 1

                if self.is_phase_blocked_cell(nb, next_step):
                    continue

                next_state = (nb, (t_mod + 1) % period)
                tentative_g = current_g + 1

                if tentative_g < g_score.get(next_state, float("inf")):
                    came_from[next_state] = state
                    g_score[next_state] = tentative_g
                    f = tentative_g + self.heuristic(nb, goal)
                    heapq.heappush(open_heap, (f, tentative_g, next_state))

        return []

    def plan_from(self, pos):
        self.path = self.astar(pos, self.goal, self.total_steps)
        self.path_index = 0
        return len(self.path) > 0

    def find_initial_path(self):
        return self.plan_from(self.start)

    def mark_visited(self, cell):
        if self.visited_mask is None:
            return
        r, c = cell
        self.visited_mask[r, c] = True

    def get_remaining_path(self):
        if not self.path:
            return []
        return self.path[self.path_index:]

    def step(self):
        if self.failed:
            return "stuck"

        if self.pending_respawn:
            return self._respawn_now()

        if self.dead:
            return "dead"

        if self.done:
            return "goal"

        if not self.path:
            ok = self.plan_from(self.position)
            if not ok:
                # Dynamic hazards may unblock later; wait in place and try again next turn.
                self.total_steps += 1
                if self.position in self.get_active_fire_cells():
                    self._trigger_death(self.position)
                    return "dead"
                return "wait"

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
            ok = self.plan_from(self.position)
            if not ok:
                self.total_steps += 1
                if self.position in self.get_active_fire_cells():
                    self._trigger_death(self.position)
                    return "dead"
                return "wait"
            return "move"

        self.path_index += 1
        self.position = actual_next

        if self.position != self.visited[-1]:
            self.visited.append(self.position)
        self.mark_visited(self.position)
        self.mark_visited(self.position)
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
            self.mark_visited(self.position)
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
                self.total_steps += 1
                if self.position in self.get_active_fire_cells():
                    self._trigger_death(self.position)
                    return "dead"
                return "wait"

            return event

        return event