"""
Intersection Geometry, Lane Definitions, Trajectory Splines, and Traffic Light Poles.
Builds cubic Bezier paths with arc-length parameterization for smooth turn physics.
"""
import math
import numpy as np
from src.config import (
    CENTER_X, CENTER_Y, ROAD_WIDTH, LANE_WIDTH,
    SCREEN_WIDTH, SCREEN_HEIGHT, STOP_LINE_OFFSET
)

def cubic_bezier(p0, p1, p2, p3, t):
    """Calculate point on cubic Bezier curve at parameter t in [0, 1]."""
    u = 1.0 - t
    tt = t * t
    uu = u * u
    uuu = uu * u
    ttt = tt * t

    x = uuu * p0[0] + 3 * uu * t * p1[0] + 3 * u * tt * p2[0] + ttt * p3[0]
    y = uuu * p0[1] + 3 * uu * t * p1[1] + 3 * u * tt * p2[1] + ttt * p3[1]
    return (x, y)

def cubic_bezier_tangent(p0, p1, p2, p3, t):
    """Calculate tangent vector on cubic Bezier curve at parameter t."""
    u = 1.0 - t
    dx = 3 * u * u * (p1[0] - p0[0]) + 6 * u * t * (p2[0] - p1[0]) + 3 * t * t * (p3[0] - p2[0])
    dy = 3 * u * u * (p1[1] - p0[1]) + 6 * u * t * (p2[1] - p1[1]) + 3 * t * t * (p3[1] - p2[1])
    return (dx, dy)


class Route:
    def __init__(self, route_id, start_dir, end_dir, turn_type, key_points, stop_line_dist):
        self.id = route_id
        self.start_dir = start_dir   # 'N', 'S', 'E', 'W'
        self.end_dir = end_dir       # 'N', 'S', 'E', 'W'
        self.turn_type = turn_type   # 'STRAIGHT', 'LEFT', 'RIGHT'
        self.key_points = key_points
        self.stop_line_dist = stop_line_dist

        # Sample and build uniform arc-length parameterization table
        self.samples = []
        self.cumulative_dist = [0.0]
        self._build_arc_length_table(num_steps=320)
        self.total_length = self.cumulative_dist[-1]

    def _build_arc_length_table(self, num_steps=320):
        # Generate piecewise high-density coordinate waypoints
        raw_pts = []
        if self.turn_type == 'STRAIGHT':
            p_start, p_end = self.key_points[0], self.key_points[-1]
            for i in range(num_steps + 1):
                t = i / float(num_steps)
                x = p_start[0] + (p_end[0] - p_start[0]) * t
                y = p_start[1] + (p_end[1] - p_start[1]) * t
                raw_pts.append((x, y))
        else:
            p_start, p_enter, p_exit, p_end = self.key_points
            # 1. Approach straight
            approach_steps = int(num_steps * 0.30)
            for i in range(approach_steps):
                t = i / float(approach_steps)
                x = p_start[0] + (p_enter[0] - p_start[0]) * t
                y = p_start[1] + (p_enter[1] - p_start[1]) * t
                raw_pts.append((x, y))

            # 2. Smooth cubic turn curve inside the intersection box
            turn_steps = int(num_steps * 0.40)
            d = math.hypot(p_exit[0] - p_enter[0], p_exit[1] - p_enter[1]) * 0.45
            v_in = (p_enter[0] - p_start[0], p_enter[1] - p_start[1])
            len_in = math.hypot(*v_in)
            tan_in = (v_in[0]/len_in, v_in[1]/len_in) if len_in > 0 else (0, 1)

            v_out = (p_end[0] - p_exit[0], p_end[1] - p_exit[1])
            len_out = math.hypot(*v_out)
            tan_out = (v_out[0]/len_out, v_out[1]/len_out) if len_out > 0 else (1, 0)

            p1 = (p_enter[0] + tan_in[0] * d, p_enter[1] + tan_in[1] * d)
            p2 = (p_exit[0] - tan_out[0] * d, p_exit[1] - tan_out[1] * d)

            for i in range(turn_steps):
                t = i / float(turn_steps)
                u = 1.0 - t
                x = u**3 * p_enter[0] + 3*u**2*t * p1[0] + 3*u*t**2 * p2[0] + t**3 * p_exit[0]
                y = u**3 * p_enter[1] + 3*u**2*t * p1[1] + 3*u*t**2 * p2[1] + t**3 * p_exit[1]
                raw_pts.append((x, y))

            # 3. Exit straight
            exit_steps = num_steps - approach_steps - turn_steps + 1
            for i in range(exit_steps):
                t = i / float(exit_steps - 1) if exit_steps > 1 else 1.0
                x = p_exit[0] + (p_end[0] - p_exit[0]) * t
                y = p_exit[1] + (p_end[1] - p_exit[1]) * t
                raw_pts.append((x, y))

        # Calculate arc lengths and instantaneous angles
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
        """
        Returns (x, y, angle_radians) at a given distance along route.
        Returns None if distance exceeds route length.
        """
        if distance > self.total_length:
            return None
        if distance <= 0.0:
            return self.samples[0]

        # Binary search for segment in cumulative_dist
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

        # Angle interpolation with wrap around
        a0 = p0[2]
        a1 = p1[2]
        diff = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        angle = a0 + diff * ratio

        return (x, y, angle)


class Intersection:
    def __init__(self, cx=None, cy=None, width=None, height=None, topnav_h=48, sidebar_w=340):
        self.road_width = ROAD_WIDTH
        self.half_rw = ROAD_WIDTH / 2.0
        self.lane_w = LANE_WIDTH
        self.topnav_h = topnav_h
        self.sidebar_w = sidebar_w
        self.width = width or SCREEN_WIDTH
        self.height = height or SCREEN_HEIGHT

        if cx is None or cy is None:
            canvas_w = self.width - self.sidebar_w
            canvas_h = self.height - self.topnav_h
            self.cx = canvas_w // 2
            self.cy = self.topnav_h + (canvas_h // 2)
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
        self._recalculate_geometry()

    def _recalculate_geometry(self):
        # Conflict junction bounding box
        self.junction_bounds = (
            self.cx - self.half_rw,
            self.cy - self.half_rw,
            self.cx + self.half_rw,
            self.cy + self.half_rw
        )

        # Traffic light pole positions (screen coordinates)
        offset = self.half_rw + 18
        self.light_poles = {
            'N': (self.cx - self.half_rw - 8, self.cy - offset),
            'S': (self.cx + self.half_rw + 8, self.cy + offset),
            'E': (self.cx + offset, self.cy - self.half_rw - 8),
            'W': (self.cx - offset, self.cy + self.half_rw + 8),
        }

        # Stop lines (x, y, is_horizontal) - positioned before the crosswalks
        stop_offset = 36
        self.stop_lines = {
            'N': (self.cx - self.lane_w, self.cy - self.half_rw - stop_offset),
            'S': (self.cx + self.lane_w, self.cy + self.half_rw + stop_offset),
            'E': (self.cx + self.half_rw + stop_offset, self.cy - self.lane_w),
            'W': (self.cx - self.half_rw - stop_offset, self.cy + self.lane_w),
        }

        # Build all allowable traffic routes
        self.routes = self._generate_all_routes()

    def _generate_all_routes(self):
        """Builds all 12 valid routes across the 4-way intersection (100% on asphalt)."""
        routes = []
        r_id = 0

        cx, cy = self.cx, self.cy
        rw = self.half_rw # 70
        lw = self.lane_w  # 35

        canvas_right = self.width - self.sidebar_w
        canvas_top = self.topnav_h
        canvas_bottom = self.height

        # Spawn coordinates
        n_spawn_y = canvas_top - 35
        s_spawn_y = canvas_bottom + 35
        w_spawn_x = -35
        e_spawn_x = canvas_right + 35

        stop_offset = 36
        n_stop_dist = (cy - rw - stop_offset) - n_spawn_y
        s_stop_dist = s_spawn_y - (cy + rw + stop_offset)
        w_stop_dist = (cx - rw - stop_offset) - w_spawn_x
        e_stop_dist = e_spawn_x - (cx + rw + stop_offset)

        # 1. SOUTHBOUND (Coming from North, heading South / West / East)
        routes.append(Route(r_id, 'N', 'S', 'STRAIGHT', 
            [(cx - 1.5*lw, n_spawn_y), (cx - 1.5*lw, canvas_bottom + 45)], n_stop_dist))
        r_id += 1

        routes.append(Route(r_id, 'N', 'W', 'RIGHT', 
            [(cx - 1.5*lw, n_spawn_y), (cx - 1.5*lw, cy - rw), (cx - rw, cy - 1.5*lw), (-45, cy - 1.5*lw)], n_stop_dist))
        r_id += 1

        routes.append(Route(r_id, 'N', 'E', 'LEFT', 
            [(cx - 0.5*lw, n_spawn_y), (cx - 0.5*lw, cy - rw), (cx + rw, cy + 0.5*lw), (canvas_right + 45, cy + 0.5*lw)], n_stop_dist))
        r_id += 1

        # 2. NORTHBOUND (Coming from South, heading North / East / West)
        routes.append(Route(r_id, 'S', 'N', 'STRAIGHT', 
            [(cx + 1.5*lw, s_spawn_y), (cx + 1.5*lw, canvas_top - 45)], s_stop_dist))
        r_id += 1

        routes.append(Route(r_id, 'S', 'E', 'RIGHT', 
            [(cx + 1.5*lw, s_spawn_y), (cx + 1.5*lw, cy + rw), (cx + rw, cy + 1.5*lw), (canvas_right + 45, cy + 1.5*lw)], s_stop_dist))
        r_id += 1

        routes.append(Route(r_id, 'S', 'W', 'LEFT', 
            [(cx + 0.5*lw, s_spawn_y), (cx + 0.5*lw, cy + rw), (cx - rw, cy - 0.5*lw), (-45, cy - 0.5*lw)], s_stop_dist))
        r_id += 1

        # 3. EASTBOUND (Coming from West, heading East / South / North)
        routes.append(Route(r_id, 'W', 'E', 'STRAIGHT', 
            [(w_spawn_x, cy + 1.5*lw), (canvas_right + 45, cy + 1.5*lw)], w_stop_dist))
        r_id += 1

        routes.append(Route(r_id, 'W', 'S', 'RIGHT', 
            [(w_spawn_x, cy + 1.5*lw), (cx - rw, cy + 1.5*lw), (cx - 1.5*lw, cy + rw), (cx - 1.5*lw, canvas_bottom + 45)], w_stop_dist))
        r_id += 1

        routes.append(Route(r_id, 'W', 'N', 'LEFT', 
            [(w_spawn_x, cy + 0.5*lw), (cx - rw, cy + 0.5*lw), (cx + 0.5*lw, cy - rw), (cx + 0.5*lw, canvas_top - 45)], w_stop_dist))
        r_id += 1

        # 4. WESTBOUND (Coming from East, heading West / North / South)
        routes.append(Route(r_id, 'E', 'W', 'STRAIGHT', 
            [(e_spawn_x, cy - 1.5*lw), (-45, cy - 1.5*lw)], e_stop_dist))
        r_id += 1

        routes.append(Route(r_id, 'E', 'N', 'RIGHT', 
            [(e_spawn_x, cy - 1.5*lw), (cx + rw, cy - 1.5*lw), (cx + 1.5*lw, cy - rw), (cx + 1.5*lw, canvas_top - 45)], e_stop_dist))
        r_id += 1

        routes.append(Route(r_id, 'E', 'S', 'LEFT', 
            [(e_spawn_x, cy - 0.5*lw), (cx + rw, cy - 0.5*lw), (cx - 0.5*lw, cy + rw), (cx - 0.5*lw, canvas_bottom + 45)], e_stop_dist))
        r_id += 1

        return routes
