"""
City network geometry: dual arterials, 3-lane approaches, connecting corridor, and roundabout.
Builds arc-length parameterized polylines for intersection turns and circulating paths.
"""
import math
import numpy as np
from src.config import (
    ROAD_WIDTH, LANE_WIDTH, NUM_LANES_PER_DIR,
    SCREEN_WIDTH, SCREEN_HEIGHT,
    ROUNDABOUT_ISLAND_R, ROUNDABOUT_INNER_R, ROUNDABOUT_CIRC_R, ROUNDABOUT_OUTER_R
)


def cubic_bezier(p0, p1, p2, p3, t):
    u = 1.0 - t
    tt = t * t
    uu = u * u
    uuu = uu * u
    ttt = tt * t
    x = uuu * p0[0] + 3 * uu * t * p1[0] + 3 * u * tt * p2[0] + ttt * p3[0]
    y = uuu * p0[1] + 3 * uu * t * p1[1] + 3 * u * tt * p2[1] + ttt * p3[1]
    return (x, y)


def cubic_bezier_tangent(p0, p1, p2, p3, t):
    u = 1.0 - t
    dx = 3 * u * u * (p1[0] - p0[0]) + 6 * u * t * (p2[0] - p1[0]) + 3 * t * t * (p3[0] - p2[0])
    dy = 3 * u * u * (p1[1] - p0[1]) + 6 * u * t * (p2[1] - p1[1]) + 3 * t * t * (p3[1] - p2[1])
    return (dx, dy)


def _resample_polyline(points, num_steps):
    if len(points) < 2:
        return list(points)
    cum = [0.0]
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        cum.append(cum[-1] + math.hypot(dx, dy))
    total = cum[-1] if cum[-1] > 1e-6 else 1.0
    out = []
    j = 0
    for i in range(num_steps + 1):
        d = total * i / float(num_steps)
        while j + 1 < len(cum) and cum[j + 1] < d:
            j += 1
        j1 = min(j + 1, len(points) - 1)
        seg = cum[j1] - cum[j]
        ratio = (d - cum[j]) / seg if seg > 1e-6 else 0.0
        x = points[j][0] + (points[j1][0] - points[j][0]) * ratio
        y = points[j][1] + (points[j1][1] - points[j][1]) * ratio
        out.append((x, y))
    return out


def _arc_points(cx, cy, radius, theta0, sweep, n=48, ccw=True):
    """
    Arc in y-up polar angle (0=east, pi/2=north).
    ccw=True: north -> west -> south -> east (left turns / inner loop).
    ccw=False: north -> east -> south -> west (through on the near side).
    """
    pts = []
    sign = 1.0 if ccw else -1.0
    for i in range(n + 1):
        t = i / float(n)
        th = theta0 + sign * t * sweep
        pts.append((cx + radius * math.cos(th), cy - radius * math.sin(th)))
    return pts


class Route:
    def __init__(self, route_id, start_dir, end_dir, turn_type, key_points, stop_line_dist,
                 node='MAIN', yield_line_dist=None, lane_index=1, rbx=0.0, rby=0.0, r_outer=88.0,
                 entry_alpha=None, circ_ccw=True):
        self.id = route_id
        self.start_dir = start_dir
        self.end_dir = end_dir
        self.turn_type = turn_type
        self.key_points = key_points
        self.stop_line_dist = stop_line_dist
        self.node = node
        self.yield_line_dist = yield_line_dist
        self.lane_index = lane_index
        self.rbx = rbx
        self.rby = rby
        self.r_outer = r_outer
        # Angle at which this route joins the circulating ring, used by the
        # give-way rule to tell which traffic actually has priority over us.
        self.entry_alpha = entry_alpha
        self.circ_ccw = circ_ccw

        self.samples = []
        self.cumulative_dist = [0.0]
        self._build_arc_length_table(num_steps=360)
        self.total_length = self.cumulative_dist[-1]

    def _build_arc_length_table(self, num_steps=360):
        dense = self.turn_type in ('ROUNDABOUT', 'THROUGH', 'POLYLINE') or len(self.key_points) > 4
        if dense:
            raw_pts = _resample_polyline(self.key_points, num_steps)
        elif self.turn_type == 'STRAIGHT':
            p_start, p_end = self.key_points[0], self.key_points[-1]
            raw_pts = []
            for i in range(num_steps + 1):
                t = i / float(num_steps)
                x = p_start[0] + (p_end[0] - p_start[0]) * t
                y = p_start[1] + (p_end[1] - p_start[1]) * t
                raw_pts.append((x, y))
        else:
            p_start, p_enter, p_exit, p_end = self.key_points
            raw_pts = []
            approach_steps = int(num_steps * 0.30)
            for i in range(approach_steps):
                t = i / float(approach_steps)
                x = p_start[0] + (p_enter[0] - p_start[0]) * t
                y = p_start[1] + (p_enter[1] - p_start[1]) * t
                raw_pts.append((x, y))

            turn_steps = int(num_steps * 0.40)
            d = math.hypot(p_exit[0] - p_enter[0], p_exit[1] - p_enter[1]) * 0.45
            v_in = (p_enter[0] - p_start[0], p_enter[1] - p_start[1])
            len_in = math.hypot(*v_in)
            tan_in = (v_in[0] / len_in, v_in[1] / len_in) if len_in > 0 else (0, 1)

            v_out = (p_end[0] - p_exit[0], p_end[1] - p_exit[1])
            len_out = math.hypot(*v_out)
            tan_out = (v_out[0] / len_out, v_out[1] / len_out) if len_out > 0 else (1, 0)

            p1 = (p_enter[0] + tan_in[0] * d, p_enter[1] + tan_in[1] * d)
            p2 = (p_exit[0] - tan_out[0] * d, p_exit[1] - tan_out[1] * d)

            for i in range(turn_steps):
                t = i / float(turn_steps)
                u = 1.0 - t
                x = u**3 * p_enter[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p_exit[0]
                y = u**3 * p_enter[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p_exit[1]
                raw_pts.append((x, y))

            exit_steps = num_steps - approach_steps - turn_steps + 1
            for i in range(exit_steps):
                t = i / float(exit_steps - 1) if exit_steps > 1 else 1.0
                x = p_exit[0] + (p_end[0] - p_exit[0]) * t
                y = p_exit[1] + (p_end[1] - p_exit[1]) * t
                raw_pts.append((x, y))

        prev_pt = raw_pts[0]
        dx = raw_pts[1][0] - raw_pts[0][0]
        dy = raw_pts[1][1] - raw_pts[0][1]
        self.samples.append((prev_pt[0], prev_pt[1], math.atan2(dy, dx)))

        total_d = 0.0
        for i in range(1, len(raw_pts)):
            pt = raw_pts[i]
            dx = pt[0] - prev_pt[0]
            dy = pt[1] - prev_pt[1]
            d = math.hypot(dx, dy)
            total_d += d
            self.cumulative_dist.append(total_d)
            angle = math.atan2(dy, dx)
            self.samples.append((pt[0], pt[1], angle))
            prev_pt = pt

    def get_pose_at_distance(self, distance):
        if distance > self.total_length:
            return None
        if distance <= 0.0:
            return self.samples[0]

        idx = np.searchsorted(self.cumulative_dist, distance)
        if idx >= len(self.cumulative_dist):
            return self.samples[-1]

        idx0 = max(0, idx - 1)
        idx1 = idx
        d0 = self.cumulative_dist[idx0]
        d1 = self.cumulative_dist[idx1]
        seg_len = d1 - d0
        ratio = (distance - d0) / seg_len if seg_len > 1e-6 else 0.0

        p0 = self.samples[idx0]
        p1 = self.samples[idx1]
        x = p0[0] + (p1[0] - p0[0]) * ratio
        y = p0[1] + (p1[1] - p0[1]) * ratio
        a0 = p0[2]
        a1 = p1[2]
        diff = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        angle = a0 + diff * ratio
        return (x, y, angle)


# Entry heading alphas kept only as documentation of compass mouths.
_ENTRY_ALPHA = {'N': -math.pi / 2, 'W': math.pi, 'S': math.pi / 2, 'E': 0.0}


class Intersection:
    """Dual-node city: signalized 4-way + roundabout boulevard, 3 lanes each way."""

    def __init__(self, cx=None, cy=None, width=None, height=None, topnav_h=48, sidebar_w=340):
        self.road_width = ROAD_WIDTH
        self.half_rw = ROAD_WIDTH / 2.0
        self.lane_w = LANE_WIDTH
        self.n_lanes = NUM_LANES_PER_DIR
        self.topnav_h = topnav_h
        self.sidebar_w = sidebar_w
        self.width = width or SCREEN_WIDTH
        self.height = height or SCREEN_HEIGHT
        self.r_island = ROUNDABOUT_ISLAND_R
        self.r_inner = ROUNDABOUT_INNER_R
        self.r_circ = ROUNDABOUT_CIRC_R
        self.r_outer = ROUNDABOUT_OUTER_R

        if cx is None or cy is None:
            canvas_w = self.width - self.sidebar_w
            canvas_h = self.height - self.topnav_h
            self.cx = canvas_w // 2
            self.cy = self.topnav_h + int(canvas_h * 0.34)
        else:
            self.cx = cx
            self.cy = cy

        self._recalculate_geometry()

    def update_dimensions(self, cx, cy, width, height, topnav_h=48, sidebar_w=340):
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.topnav_h = topnav_h
        self.sidebar_w = sidebar_w
        canvas_h = self.height - self.topnav_h
        self.cy = self.topnav_h + int(canvas_h * 0.34)
        self.cx = (self.width - self.sidebar_w) // 2
        self._recalculate_geometry()

    def _lane_offset(self, from_right):
        return (self.n_lanes - 0.5 - from_right) * self.lane_w

    def _ns_x(self, start_dir, from_right):
        off = self._lane_offset(from_right)
        return self.cx - off if start_dir == 'N' else self.cx + off

    def _ew_y(self, start_dir, from_right, cy=None):
        off = self._lane_offset(from_right)
        origin = self.cy if cy is None else cy
        return origin + off if start_dir == 'W' else origin - off

    def _recalculate_geometry(self):
        canvas_h = self.height - self.topnav_h
        self.rbx = self.cx
        self.rby = self.topnav_h + int(canvas_h * 0.74)
        min_gap = self.half_rw + self.r_outer + 70
        if self.rby - self.cy < min_gap:
            self.rby = int(self.cy + min_gap)

        self.junction_bounds = (
            self.cx - self.half_rw,
            self.cy - self.half_rw,
            self.cx + self.half_rw,
            self.cy + self.half_rw
        )
        self.roundabout_bounds = (
            self.rbx - self.r_outer,
            self.rby - self.r_outer,
            self.rbx + self.r_outer,
            self.rby + self.r_outer
        )

        offset = self.half_rw + 18
        self.light_poles = {
            'N': (self.cx - self.half_rw - 8, self.cy - offset),
            'S': (self.cx + self.half_rw + 8, self.cy + offset),
            'E': (self.cx + offset, self.cy - self.half_rw - 8),
            'W': (self.cx - offset, self.cy + self.half_rw + 8),
        }

        stop_offset = 36
        self.stop_lines = {
            'N': (self.cx - self.lane_w, self.cy - self.half_rw - stop_offset),
            'S': (self.cx + self.lane_w, self.cy + self.half_rw + stop_offset),
            'E': (self.cx + self.half_rw + stop_offset, self.cy - self.lane_w),
            'W': (self.cx - self.half_rw - stop_offset, self.cy + self.lane_w),
        }
        self.routes = self._generate_all_routes()

    def bounds_for(self, car):
        """
        Conflict box the vehicle is currently inside, chosen by position rather
        than by route type so a THROUGH route gets the roundabout box once it has
        driven down there.
        """
        rbx = getattr(car.route, 'rbx', self.rbx)
        rby = getattr(car.route, 'rby', self.rby)
        r_outer = getattr(car.route, 'r_outer', self.r_outer)
        if math.hypot(car.x - rbx, car.y - rby) < r_outer + 48.0:
            return self.roundabout_bounds
        return self.junction_bounds

    def signal_for(self, car, traffic_controller):
        """
        Signal this vehicle must obey right now.

        Shared by the live simulation and the headless trainer. They previously
        derived it differently, so an identical situation was labelled RED in one
        and YIELD in the other, which poisoned the replay buffer.
        """
        node = getattr(car.route, 'node', 'MAIN')
        if node == 'ROUNDABOUT':
            return traffic_controller.get_light_state(car.route.start_dir, node='ROUNDABOUT')
        if node == 'THROUGH':
            rbx = getattr(car.route, 'rbx', self.rbx)
            rby = getattr(car.route, 'rby', self.rby)
            r_outer = getattr(car.route, 'r_outer', self.r_outer)
            if car.has_passed_intersection and math.hypot(car.x - rbx, car.y - rby) < r_outer + 55.0:
                return 'YIELD'
        return traffic_controller.get_light_state(car.route.start_dir)

    def conflict_bounds_for(self, route):
        if getattr(route, 'node', 'MAIN') in ('ROUNDABOUT', 'THROUGH'):
            if route.node == 'THROUGH':
                return self.junction_bounds
            return self.roundabout_bounds
        return self.junction_bounds

    def _polyline_length(self, pts):
        total = 0.0
        for i in range(1, len(pts)):
            total += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        return total

    def _circ_radius(self, from_right, start_dir, end_dir):
        """Through stays on the outer ring; left-lane turns can use the inner ring."""
        if start_dir != end_dir and {start_dir, end_dir} in ({'N', 'S'}, {'W', 'E'}):
            return self.r_circ
        return self.r_inner if from_right >= 2 else self.r_circ

    def _travel_axis(self, heading, from_right):
        """Lane line for a compass heading (drive-on-right)."""
        if heading == 'S':
            return 'x', self._ns_x('N', from_right)
        if heading == 'N':
            return 'x', self._ns_x('S', from_right)
        if heading == 'E':
            return 'y', self._ew_y('W', from_right, self.rby)
        return 'y', self._ew_y('E', from_right, self.rby)

    def _mouth_on_circle(self, mouth, heading, from_right, radius):
        """Intersect a travel lane with the circulating circle at a compass mouth."""
        rbx, rby = self.rbx, self.rby
        r = max(radius, 8.0)
        axis, val = self._travel_axis(heading, from_right)
        if axis == 'x':
            d = max(-r + 2.0, min(r - 2.0, val - rbx))
            x = rbx + d
            y = rby - math.sqrt(max(1.0, r * r - d * d)) if mouth == 'N' else rby + math.sqrt(max(1.0, r * r - d * d))
        else:
            d = max(-r + 2.0, min(r - 2.0, val - rby))
            y = rby + d
            x = rbx - math.sqrt(max(1.0, r * r - d * d)) if mouth == 'W' else rbx + math.sqrt(max(1.0, r * r - d * d))
        return (x, y)

    def _theta(self, pt):
        return math.atan2(self.rby - pt[1], pt[0] - self.rbx)

    def _extend_into_roundabout(self, head, start_dir, end_dir, lane, dest):
        rest, _, _ = self._roundabout_path(start_dir, end_dir, lane, head[-1], dest)
        return head + rest[1:]

    def _roundabout_path(self, start_dir, end_dir, from_right, spawn, dest):
        """
        One-way CCW ring (drive-on-right): N→W→S→E.

        Through and turning traffic share that direction so nobody meets
        oncoming cars on the circle. Exits still land on the outbound lane.
        """
        radius = self._circ_radius(from_right, start_dir, end_dir)
        arrive = {'N': 'S', 'S': 'N', 'W': 'E', 'E': 'W'}[start_dir]
        enter = self._mouth_on_circle(start_dir, arrive, from_right, radius)
        leave = self._mouth_on_circle(end_dir, end_dir, from_right, radius)
        th0 = self._theta(enter)
        th1 = self._theta(leave)
        sweep = (th1 - th0) % (2.0 * math.pi)
        if sweep < math.radians(40.0):
            sweep += 2.0 * math.pi
        pts = [spawn, enter]
        pts.extend(_arc_points(self.rbx, self.rby, radius, th0, sweep, n=42, ccw=True)[1:])
        pts.append(dest)
        return pts, th0, True

    def _generate_all_routes(self):
        routes = []
        r_id = 0
        cx, cy = self.cx, self.cy
        rw = self.half_rw
        canvas_right = self.width - self.sidebar_w
        canvas_top = self.topnav_h
        n_spawn_y = canvas_top - 35
        s_spawn_y = self.height + 35
        w_spawn_x = -35
        e_spawn_x = canvas_right + 35
        stop_offset = 36
        mid_y = (cy + self.rby) / 2.0

        def add(start, end, ttype, pts, stop_d, node='MAIN', yield_d=None, lane=1,
                entry_alpha=None, circ_ccw=True):
            nonlocal r_id
            routes.append(Route(
                r_id, start, end, ttype, pts, stop_d,
                node=node, yield_line_dist=yield_d, lane_index=lane,
                rbx=self.rbx, rby=self.rby, r_outer=self.r_outer,
                entry_alpha=entry_alpha, circ_ccw=circ_ccw
            ))
            r_id += 1

        n_stop = (cy - rw - stop_offset) - n_spawn_y
        w_stop = (cx - rw - stop_offset) - w_spawn_x
        e_stop = e_spawn_x - (cx + rw + stop_offset)
        s_local_spawn = mid_y + 55
        s_local_stop = max(40.0, s_local_spawn - (cy + rw + stop_offset))

        add('N', 'W', 'RIGHT', [
            (self._ns_x('N', 0), n_spawn_y), (self._ns_x('N', 0), cy - rw),
            (cx - rw, cy - self._lane_offset(0)), (-45, cy - self._lane_offset(0))
        ], n_stop, lane=0)
        add('N', 'E', 'LEFT', [
            (self._ns_x('N', 2), n_spawn_y), (self._ns_x('N', 2), cy - rw),
            (cx + rw, self._ew_y('W', 2)), (canvas_right + 45, self._ew_y('W', 2))
        ], n_stop, lane=2)

        thru_ns = [(self._ns_x('N', 1), n_spawn_y), (self._ns_x('N', 1), cy + rw + 8)]
        rest_ns, a_ns, ccw_ns = self._roundabout_path('N', 'S', 1, thru_ns[-1], (self._ns_x('N', 1), s_spawn_y))
        thru_ns.extend(rest_ns[1:])
        add('N', 'S', 'THROUGH', thru_ns, n_stop, node='THROUGH',
            yield_d=self._polyline_length(thru_ns[:2]) + 40, lane=1,
            entry_alpha=a_ns, circ_ccw=ccw_ns)

        for lane in (0, 2):
            head = [(self._ns_x('N', lane), n_spawn_y), (self._ns_x('N', lane), cy + rw + 8)]
            rest, a_n, ccw_n = self._roundabout_path('N', 'S', lane, head[-1], (self._ns_x('N', lane), s_spawn_y))
            pts = head + rest[1:]
            add('N', 'S', 'THROUGH', pts, n_stop, node='THROUGH',
                yield_d=self._polyline_length(pts[:2]) + 40, lane=lane,
                entry_alpha=a_n, circ_ccw=ccw_n)

        add('S', 'N', 'STRAIGHT', [
            (self._ns_x('S', 1), s_local_spawn), (self._ns_x('S', 1), canvas_top - 45)
        ], s_local_stop, lane=1)
        add('S', 'E', 'RIGHT', [
            (self._ns_x('S', 0), s_local_spawn), (self._ns_x('S', 0), cy + rw),
            (cx + rw, cy + self._lane_offset(0)), (canvas_right + 45, cy + self._lane_offset(0))
        ], s_local_stop, lane=0)
        add('S', 'W', 'LEFT', [
            (self._ns_x('S', 2), s_local_spawn), (self._ns_x('S', 2), cy + rw),
            (cx - rw, self._ew_y('E', 2)), (-45, self._ew_y('E', 2))
        ], s_local_stop, lane=2)

        add('W', 'E', 'STRAIGHT', [
            (w_spawn_x, self._ew_y('W', 1)), (canvas_right + 45, self._ew_y('W', 1))
        ], w_stop, lane=1)
        # Right at the 4-way, then continue into the circle and exit south.
        # Ending at the north mouth used to dump a queue on top of the yield line.
        ws_head = [
            (w_spawn_x, self._ew_y('W', 0)),
            (cx - rw, self._ew_y('W', 0)),
            (self._ns_x('N', 0), self._ew_y('W', 0)),
            (self._ns_x('N', 0), cy + rw),
        ]
        ws_pts = self._extend_into_roundabout(ws_head, 'N', 'S', 0, (self._ns_x('N', 0), s_spawn_y))
        _, a_ws, ccw_ws = self._roundabout_path('N', 'S', 0, ws_head[-1], (self._ns_x('N', 0), s_spawn_y))
        add('W', 'S', 'THROUGH', ws_pts, w_stop, node='THROUGH',
            yield_d=self._polyline_length(ws_pts[:5]) - 8.0, lane=0,
            entry_alpha=a_ws, circ_ccw=ccw_ws)
        add('W', 'N', 'LEFT', [
            (w_spawn_x, self._ew_y('W', 2)), (cx - rw, self._ew_y('W', 2)),
            (self._ns_x('S', 2), cy - rw), (self._ns_x('S', 2), canvas_top - 45)
        ], w_stop, lane=2)

        add('E', 'W', 'STRAIGHT', [
            (e_spawn_x, self._ew_y('E', 1)), (-45, self._ew_y('E', 1))
        ], e_stop, lane=1)
        add('E', 'N', 'RIGHT', [
            (e_spawn_x, self._ew_y('E', 0)), (cx + rw, self._ew_y('E', 0)),
            (self._ns_x('S', 0), cy - rw), (self._ns_x('S', 0), canvas_top - 45)
        ], e_stop, lane=0)
        es_head = [
            (e_spawn_x, self._ew_y('E', 2)),
            (cx + rw, self._ew_y('E', 2)),
            (self._ns_x('N', 2), self._ew_y('E', 2)),
            (self._ns_x('N', 2), cy + rw),
        ]
        es_pts = self._extend_into_roundabout(es_head, 'N', 'S', 2, (self._ns_x('N', 2), s_spawn_y))
        _, a_es, ccw_es = self._roundabout_path('N', 'S', 2, es_head[-1], (self._ns_x('N', 2), s_spawn_y))
        add('E', 'S', 'THROUGH', es_pts, e_stop, node='THROUGH',
            yield_d=self._polyline_length(es_pts[:5]) - 8.0, lane=2,
            entry_alpha=a_es, circ_ccw=ccw_es)

        def blvd_spawn(d, lane=1):
            if d == 'W':
                return (w_spawn_x, self._ew_y('W', lane, self.rby))
            if d == 'E':
                return (e_spawn_x, self._ew_y('E', lane, self.rby))
            return (self._ns_x('S', lane), s_spawn_y)

        def blvd_dest(d, lane=1):
            if d == 'W':
                return (-45, self._ew_y('E', lane, self.rby))
            if d == 'E':
                return (canvas_right + 45, self._ew_y('W', lane, self.rby))
            if d == 'S':
                return (self._ns_x('N', lane), s_spawn_y)
            return (self._ns_x('S', lane), canvas_top - 45)

        for start, end in (
            ('W', 'S'), ('W', 'N'), ('W', 'E'),
            ('E', 'N'), ('E', 'S'), ('E', 'W'),
            ('S', 'E'), ('S', 'W'),
        ):
            for lane in (0, 1, 2):
                spawn = blvd_spawn(start, lane)
                pts, a0, ccw = self._roundabout_path(start, end, lane, spawn, blvd_dest(end, lane))
                yield_d = max(30.0, math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]) - 8)
                add(start, end, 'ROUNDABOUT', pts, 1e6, node='ROUNDABOUT', yield_d=yield_d, lane=lane,
                    entry_alpha=a0, circ_ccw=ccw)

        for lane in (0, 1, 2):
            spawn_s = blvd_spawn('S', lane)
            pts_sn, a_sn, ccw_sn = self._roundabout_path('S', 'N', lane, spawn_s, (self._ns_x('S', lane), cy + rw + 12))
            pts_sn.append((self._ns_x('S', lane), canvas_top - 45))
            yield_sn = max(30.0, math.hypot(pts_sn[1][0] - pts_sn[0][0], pts_sn[1][1] - pts_sn[0][1]) - 8)
            stop_sn = self._polyline_length(pts_sn) * 0.62
            add('S', 'N', 'THROUGH', pts_sn, stop_sn, node='THROUGH', yield_d=yield_sn, lane=lane,
                entry_alpha=a_sn, circ_ccw=ccw_sn)

        for end in ('E', 'W'):
            for lane in (0, 1, 2):
                spawn = (self._ns_x('N', lane), n_spawn_y)
                head = [(spawn[0], spawn[1]), (spawn[0], cy + rw + 8)]
                rest, a_n, ccw_n = self._roundabout_path('N', end, lane, head[-1], blvd_dest(end, lane))
                pts = head + rest[1:]
                add('N', end, 'THROUGH', pts, n_stop, node='THROUGH',
                    yield_d=self._polyline_length(pts[:3]), lane=lane,
                    entry_alpha=a_n, circ_ccw=ccw_n)

        return routes
