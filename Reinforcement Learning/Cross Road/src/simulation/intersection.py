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
    def __init__(self, route_id, start_dir, end_dir, turn_type, control_points, stop_line_dist):
        self.id = route_id
        self.start_dir = start_dir   # 'N', 'S', 'E', 'W'
        self.end_dir = end_dir       # 'N', 'S', 'E', 'W'
        self.turn_type = turn_type   # 'STRAIGHT', 'LEFT', 'RIGHT'
        self.control_points = control_points
        self.stop_line_dist = stop_line_dist

        # Sample and build uniform arc-length parameterization table
        self.samples = []
        self.cumulative_dist = [0.0]
        self._build_arc_length_table(num_steps=300)
        self.total_length = self.cumulative_dist[-1]

    def _build_arc_length_table(self, num_steps=300):
        p0, p1, p2, p3 = self.control_points
        prev_pt = cubic_bezier(p0, p1, p2, p3, 0.0)
        prev_tan = cubic_bezier_tangent(p0, p1, p2, p3, 0.0)
        angle = math.atan2(prev_tan[1], prev_tan[0])
        self.samples.append((prev_pt[0], prev_pt[1], angle))

        total_d = 0.0
        for i in range(1, num_steps + 1):
            t = i / num_steps
            pt = cubic_bezier(p0, p1, p2, p3, t)
            tan = cubic_bezier_tangent(p0, p1, p2, p3, t)
            angle = math.atan2(tan[1], tan[0])
            d = math.hypot(pt[0] - prev_pt[0], pt[1] - prev_pt[1])
            total_d += d
            self.cumulative_dist.append(total_d)
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

        # Angle interpolation
        a0 = p0[2]
        a1 = p1[2]
        # Handle wrap around
        diff = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        angle = a0 + diff * ratio

        return (x, y, angle)


class Intersection:
    def __init__(self):
        self.cx = CENTER_X
        self.cy = CENTER_Y
        self.road_width = ROAD_WIDTH
        self.half_rw = ROAD_WIDTH / 2.0
        self.lane_w = LANE_WIDTH

        # Conflict junction bounding box
        self.junction_bounds = (
            self.cx - self.half_rw,
            self.cy - self.half_rw,
            self.cx + self.half_rw,
            self.cy + self.half_rw
        )

        # Traffic light pole positions (screen coordinates)
        # Positioned at corners of the intersection
        offset = self.half_rw + 18
        self.light_poles = {
            'N': (self.cx + self.half_rw + 8, self.cy - offset),      # Northbound approach light
            'S': (self.cx - self.half_rw - 8, self.cy + offset),      # Southbound approach light
            'E': (self.cx + offset, self.cy + self.half_rw + 8),      # Eastbound approach light
            'W': (self.cx - offset, self.cy - self.half_rw - 8),      # Westbound approach light
        }

        # Stop lines (x, y, is_horizontal)
        self.stop_lines = {
            'N': (self.cx + self.lane_w, self.cy - self.half_rw),     # Coming from North going South
            'S': (self.cx - self.lane_w, self.cy + self.half_rw),     # Coming from South going North
            'E': (self.cx + self.half_rw, self.cy - self.lane_w),     # Coming from East going West
            'W': (self.cx - self.half_rw, self.cy + self.lane_w),     # Coming from West going East
        }

        # Build all allowable traffic routes
        self.routes = self._generate_all_routes()

    def _generate_all_routes(self):
        """Builds all 12 valid routes across the 4-way intersection."""
        routes = []
        r_id = 0

        cx, cy = self.cx, self.cy
        rw = self.half_rw # 70
        lw = self.lane_w  # 35

        # Lane offsets from center:
        # Incoming Right lane: +1.5 * lw or -1.5 * lw
        # Incoming Left lane:  +0.5 * lw or -0.5 * lw
        # Outgoing lanes: opposite signs

        # 1. SOUTHBOUND (Coming from North, heading South / East / West)
        # Start points:
        n_spawn_y = -40
        n_stop_dist = (cy - rw) - n_spawn_y

        # North -> South (Straight from right lane)
        p0 = (cx + 1.5 * lw, n_spawn_y)
        p3 = (cx + 1.5 * lw, SCREEN_HEIGHT + 40)
        p1 = (cx + 1.5 * lw, cy - rw)
        p2 = (cx + 1.5 * lw, cy + rw)
        routes.append(Route(r_id, 'N', 'S', 'STRAIGHT', (p0, p1, p2, p3), n_stop_dist))
        r_id += 1

        # North -> West (Right Turn from right lane)
        p0 = (cx + 1.5 * lw, n_spawn_y)
        p3 = (-40, cy - 1.5 * lw)
        p1 = (cx + 1.5 * lw, cy - rw)
        p2 = (cx + rw, cy - 1.5 * lw)
        routes.append(Route(r_id, 'N', 'W', 'RIGHT', (p0, p1, p2, p3), n_stop_dist))
        r_id += 1

        # North -> East (Left Turn from left lane)
        p0 = (cx + 0.5 * lw, n_spawn_y)
        p3 = (SCREEN_WIDTH + 40, cy + 0.5 * lw)
        p1 = (cx + 0.5 * lw, cy + 0.5 * lw)
        p2 = (cx + rw, cy + 0.5 * lw)
        routes.append(Route(r_id, 'N', 'E', 'LEFT', (p0, p1, p2, p3), n_stop_dist))
        r_id += 1

        # 2. NORTHBOUND (Coming from South, heading North / West / East)
        s_spawn_y = SCREEN_HEIGHT + 40
        s_stop_dist = s_spawn_y - (cy + rw)

        # South -> North (Straight)
        p0 = (cx - 1.5 * lw, s_spawn_y)
        p3 = (cx - 1.5 * lw, -40)
        p1 = (cx - 1.5 * lw, cy + rw)
        p2 = (cx - 1.5 * lw, cy - rw)
        routes.append(Route(r_id, 'S', 'N', 'STRAIGHT', (p0, p1, p2, p3), s_stop_dist))
        r_id += 1

        # South -> East (Right Turn)
        p0 = (cx - 1.5 * lw, s_spawn_y)
        p3 = (SCREEN_WIDTH + 40, cy + 1.5 * lw)
        p1 = (cx - 1.5 * lw, cy + rw)
        p2 = (cx - rw, cy + 1.5 * lw)
        routes.append(Route(r_id, 'S', 'E', 'RIGHT', (p0, p1, p2, p3), s_stop_dist))
        r_id += 1

        # South -> West (Left Turn)
        p0 = (cx - 0.5 * lw, s_spawn_y)
        p3 = (-40, cy - 0.5 * lw)
        p1 = (cx - 0.5 * lw, cy - 0.5 * lw)
        p2 = (cx - rw, cy - 0.5 * lw)
        routes.append(Route(r_id, 'S', 'W', 'LEFT', (p0, p1, p2, p3), s_stop_dist))
        r_id += 1

        # 3. EASTBOUND (Coming from West, heading East / South / North)
        w_spawn_x = -40
        w_stop_dist = (cx - rw) - w_spawn_x

        # West -> East (Straight)
        p0 = (w_spawn_x, cy + 1.5 * lw)
        p3 = (SCREEN_WIDTH + 40, cy + 1.5 * lw)
        p1 = (cx - rw, cy + 1.5 * lw)
        p2 = (cx + rw, cy + 1.5 * lw)
        routes.append(Route(r_id, 'W', 'E', 'STRAIGHT', (p0, p1, p2, p3), w_stop_dist))
        r_id += 1

        # West -> South (Right Turn)
        p0 = (w_spawn_x, cy + 1.5 * lw)
        p3 = (cx + 1.5 * lw, SCREEN_HEIGHT + 40)
        p1 = (cx - rw, cy + 1.5 * lw)
        p2 = (cx + 1.5 * lw, cy + rw)
        routes.append(Route(r_id, 'W', 'S', 'RIGHT', (p0, p1, p2, p3), w_stop_dist))
        r_id += 1

        # West -> North (Left Turn)
        p0 = (w_spawn_x, cy + 0.5 * lw)
        p3 = (cx - 0.5 * lw, -40)
        p1 = (cx - 0.5 * lw, cy + 0.5 * lw)
        p2 = (cx - 0.5 * lw, cy - rw)
        routes.append(Route(r_id, 'W', 'N', 'LEFT', (p0, p1, p2, p3), w_stop_dist))
        r_id += 1

        # 4. WESTBOUND (Coming from East, heading West / North / South)
        e_spawn_x = SCREEN_WIDTH + 40
        e_stop_dist = e_spawn_x - (cx + rw)

        # East -> West (Straight)
        p0 = (e_spawn_x, cy - 1.5 * lw)
        p3 = (-40, cy - 1.5 * lw)
        p1 = (cx + rw, cy - 1.5 * lw)
        p2 = (cx - rw, cy - 1.5 * lw)
        routes.append(Route(r_id, 'E', 'W', 'STRAIGHT', (p0, p1, p2, p3), e_stop_dist))
        r_id += 1

        # East -> North (Right Turn)
        p0 = (e_spawn_x, cy - 1.5 * lw)
        p3 = (cx - 1.5 * lw, -40)
        p1 = (cx + rw, cy - 1.5 * lw)
        p2 = (cx - 1.5 * lw, cy - rw)
        routes.append(Route(r_id, 'E', 'N', 'RIGHT', (p0, p1, p2, p3), e_stop_dist))
        r_id += 1

        # East -> South (Left Turn)
        p0 = (e_spawn_x, cy - 0.5 * lw)
        p3 = (cx + 0.5 * lw, SCREEN_HEIGHT + 40)
        p1 = (cx + 0.5 * lw, cy - 0.5 * lw)
        p2 = (cx + 0.5 * lw, cy + rw)
        routes.append(Route(r_id, 'E', 'S', 'LEFT', (p0, p1, p2, p3), e_stop_dist))
        r_id += 1

        return routes
