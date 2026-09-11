"""
Perception and Computer Vision / LiDAR Sensing System for Autonomous Vehicles.
Performs 9-ray directional raycasting, Time-to-Collision (TTC) estimation,
traffic light detection, relative speed estimation, and conflict zone radar.
"""
import math
import numpy as np
from src.config import (
    LIDAR_NUM_RAYS, LIDAR_MAX_DIST, LIDAR_FOV, NUM_LANES_PER_DIR, SIREN_RADIUS,
    VISION_STATE_SIZE
)

def ray_segment_intersection(ray_origin, ray_dir, max_len, seg_p1, seg_p2):
    """
    Computes intersection of ray (origin + t * dir, 0 <= t <= max_len)
    with 2D line segment (p1 -> p2).
    Returns distance t if intersects, else None.
    """
    ox, oy = ray_origin
    dx, dy = ray_dir
    x1, y1 = seg_p1
    x2, y2 = seg_p2

    denom = dx * (y2 - y1) - dy * (x2 - x1)
    if abs(denom) < 1e-8:
        return None

    t = ((x1 - ox) * (y2 - y1) - (y1 - oy) * (x2 - x1)) / denom
    u = ((x1 - ox) * dy - (y1 - oy) * dx) / denom

    if 0 <= t <= max_len and 0 <= u <= 1.0:
        return t
    return None

class SensorSuite:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.num_rays = 9 # standard 9 rays
        self.max_dist = LIDAR_MAX_DIST
        self.fov = LIDAR_FOV
        self.ray_hits = []
        self.min_ttc = 99.0 # Time-to-collision seconds

    def update(self, all_vehicles, traffic_light_state, junction_bounds, friction_coeff=1.0, pedestrians=None):
        """
        Compute full sensory state vector for Deep RL.
        """
        car = self.vehicle
        self.ray_hits.clear()

        # Generate Ray angles relative to vehicle heading
        half_fov = self.fov / 2.0
        angle_step = self.fov / (self.num_rays - 1) if self.num_rays > 1 else 0.0

        ray_distances = []
        ray_rel_speeds = []

        front_x = car.x + math.cos(car.angle) * (car.length / 2.0)
        front_y = car.y + math.sin(car.angle) * (car.length / 2.0)
        origin = (front_x, front_y)

        # Collect segment edges of nearby active vehicles and pedestrians (spatial culling)
        other_segments = []
        max_dist_sq = (self.max_dist + 45.0) ** 2

        for other in all_vehicles:
            if other.id == car.id:
                continue
            dx = other.x - front_x
            dy = other.y - front_y
            if dx * dx + dy * dy > max_dist_sq:
                continue
            corners = other.get_corners()
            for i in range(4):
                p1 = corners[i]
                p2 = corners[(i + 1) % 4]
                other_segments.append((p1, p2, other))

        if pedestrians:
            for ped in pedestrians:
                if not ped.is_alive:
                    continue
                dx = ped.x - front_x
                dy = ped.y - front_y
                if dx * dx + dy * dy > max_dist_sq:
                    continue
                p_corners = ped.get_corners()
                for i in range(4):
                    p1 = p_corners[i]
                    p2 = p_corners[(i + 1) % 4]
                    other_segments.append((p1, p2, ped))

        min_ttc_val = 99.0
        for i in range(self.num_rays):
            rel_angle = -half_fov + i * angle_step
            ray_angle = car.angle + rel_angle
            ray_dir = (math.cos(ray_angle), math.sin(ray_angle))

            closest_t = self.max_dist
            closest_car = None

            for p1, p2, other in other_segments:
                t = ray_segment_intersection(origin, ray_dir, self.max_dist, p1, p2)
                if t is not None and t < closest_t:
                    closest_t = t
                    closest_car = other

            hit_point = (
                origin[0] + ray_dir[0] * closest_t,
                origin[1] + ray_dir[1] * closest_t
            )
            self.ray_hits.append((hit_point, closest_t, closest_car))

            norm_dist = closest_t / self.max_dist
            ray_distances.append(norm_dist)

            # Relative velocity & Time-to-Collision (TTC)
            if closest_car is not None:
                # If it's a pedestrian, their lateral walk speed doesn't decrease our forward closing speed
                is_pedestrian = hasattr(closest_car, 'walk_timer')
                other_speed = 0.0 if is_pedestrian else getattr(closest_car, 'speed', 0.0)
                closing_speed = car.speed - other_speed
                rel_v = (car.speed - other_speed) / 6.5 # Positive = closing in
                if closing_speed > 0.1:
                    ttc = (closest_t / (closing_speed * 60.0))
                    min_ttc_val = min(min_ttc_val, ttc)
            else:
                rel_v = 0.0
            ray_rel_speeds.append(rel_v)

        self.min_ttc = min_ttc_val

        # 2. Traffic Light Perception
        dist_to_stop = car.get_distance_to_stop_line()
        # Roundabout routes carry a sentinel stop line (1e6); treat it as absent.
        if dist_to_stop is not None and dist_to_stop > 4000.0:
            dist_to_stop = None
        is_before_stop = dist_to_stop is not None and dist_to_stop >= -15.0

        if is_before_stop:
            tl_state = traffic_light_state
            norm_stop_dist = max(0.0, min(1.0, dist_to_stop / 250.0)) if dist_to_stop > 0 else 0.0
        else:
            tl_state = traffic_light_state if traffic_light_state == 'YIELD' else 'NONE'
            norm_stop_dist = 1.0

        # YIELD gets its own channel; without it the roundabout produced an
        # all-zero one-hot, which is a state the network never sees anywhere else.
        tl_one_hot = [
            1.0 if tl_state == 'RED' else 0.0,
            1.0 if tl_state == 'YELLOW' else 0.0,
            1.0 if tl_state == 'GREEN' else 0.0,
            1.0 if tl_state == 'YIELD' else 0.0,
            1.0 if tl_state not in ('RED', 'YELLOW', 'GREEN', 'YIELD') else 0.0,
        ]

        # 3. Leading vehicle along the same route
        lead_dist, lead_car = car.get_leading_car_info(all_vehicles)
        norm_lead_dist = min(1.0, lead_dist / 200.0) if lead_dist is not None else 1.0

        # 4. Conflict Zone Radar
        jx_min, jy_min, jx_max, jy_max = junction_bounds
        cars_in_junction = 0
        crossing_hazard = 0.0
        for other in all_vehicles:
            if other.id != car.id and other.is_alive:
                if jx_min <= other.x <= jx_max and jy_min <= other.y <= jy_max:
                    cars_in_junction += 1
                    angle_diff = abs((other.angle - car.angle + math.pi) % (2 * math.pi) - math.pi)
                    if math.radians(45) <= angle_diff <= math.radians(135):
                        crossing_hazard = 1.0

        norm_junction_density = min(1.0, cars_in_junction / 6.0)

        # 5. Speed telemetry
        norm_speed = min(1.0, car.speed / 6.5)
        norm_target_speed = min(1.0, car.target_speed / 6.5)

        # 6. City-network extras (roundabout, emergency, lane)
        rbx = getattr(car.route, 'rbx', car.x)
        rby = getattr(car.route, 'rby', car.y)
        r_outer = getattr(car.route, 'r_outer', 88.0)
        dist_center = math.hypot(car.x - rbx, car.y - rby)
        in_roundabout = 1.0 if dist_center < r_outer + 12.0 else 0.0
        circ = 0
        emergency_prox = 0.0
        for other in all_vehicles:
            if other.id == car.id:
                continue
            dcirc = math.hypot(other.x - rbx, other.y - rby)
            if other.is_alive and 28.0 < dcirc < r_outer + 6.0:
                circ += 1
            if other.is_alive and getattr(other, 'is_emergency', False):
                ed = math.hypot(car.x - other.x, car.y - other.y)
                emergency_prox = max(emergency_prox, max(0.0, 1.0 - ed / SIREN_RADIUS))
        circ_density = min(1.0, circ / 6.0)
        lane_norm = float(getattr(car.route, 'lane_index', 1)) / max(1, NUM_LANES_PER_DIR - 1)
        yld = getattr(car.route, 'yield_line_dist', None)
        yield_req = 0.0
        norm_yield_dist = 1.0
        if yld is not None and not getattr(car, 'has_cleared_yield', False):
            dist_y = yld - car.path_distance
            norm_yield_dist = max(0.0, min(1.0, dist_y / 150.0))
            if -5.0 < dist_y < 70.0 and circ > 0:
                yield_req = 1.0

        # 7. Conflict-box occupancy of this vehicle and inverted TTC urgency
        in_box = 1.0 if (jx_min <= car.x <= jx_max and jy_min <= car.y <= jy_max) else 0.0
        ttc_urgency = max(0.0, min(1.0, 1.0 - (min_ttc_val / 5.0)))

        situation = self._situation_features(
            car, all_vehicles, traffic_light_state, lead_dist, lead_car,
            dist_center, r_outer, yield_req, in_roundabout
        )

        # Assemble the full state vector
        state = np.array(
            ray_distances +
            ray_rel_speeds +
            tl_one_hot +
            [norm_stop_dist] +
            [norm_speed] +
            [norm_target_speed] +
            [norm_lead_dist] +
            [norm_junction_density, crossing_hazard] +
            [friction_coeff] +
            [in_roundabout, circ_density, emergency_prox, lane_norm, yield_req] +
            [norm_yield_dist, in_box, ttc_urgency] +
            situation,
            dtype=np.float32
        )
        if state.shape[0] != VISION_STATE_SIZE:
            raise RuntimeError(
                f"observation size {state.shape[0]} != VISION_STATE_SIZE {VISION_STATE_SIZE}"
            )
        return state

    def _situation_features(self, car, all_vehicles, traffic_light_state, lead_dist,
                            lead_car, dist_center, r_outer, yield_req, in_roundabout):
        """
        Multi-context semantics: where this vehicle is, what the leader is doing,
        and whether the current rule set actually allows it to go.
        """
        ctx_signal = 1.0 if (
            not car.has_passed_intersection
            and getattr(car.route, 'node', 'MAIN') in ('MAIN', 'THROUGH')
            and in_roundabout < 0.5
        ) else 0.0
        ctx_ring = 1.0 if in_roundabout > 0.5 else 0.0
        ctx_entry = 1.0 if yield_req > 0.5 else 0.0
        ctx_exit = 1.0 if getattr(car, 'was_in_roundabout', False) and ctx_ring < 0.5 else 0.0
        ctx_follow = 1.0 if lead_dist is not None and lead_dist < 80.0 else 0.0

        self_ems = 1.0 if car.is_emergency else 0.0

        lead_stalled = 0.0
        lead_crossing = 0.0
        if lead_car is not None and not hasattr(lead_car, 'walk_timer'):
            if getattr(lead_car, 'speed', 1.0) < 0.25 and getattr(lead_car, 'time_stalled', 0.0) > 1.0:
                lead_stalled = 1.0
            heading_gap = abs((lead_car.angle - car.angle + math.pi) % (2 * math.pi) - math.pi)
            if math.radians(50) <= heading_gap <= math.radians(130):
                lead_crossing = 1.0

        path_clear = 1.0 if lead_dist is None or lead_dist > 90.0 else 0.0

        legal_go = 1.0
        dist_to_stop = car.get_distance_to_stop_line()
        if dist_to_stop is not None and dist_to_stop > 4000:
            dist_to_stop = None
        if (traffic_light_state in ('RED', 'YELLOW') and not car.has_passed_intersection
                and not car.is_emergency and dist_to_stop is not None and dist_to_stop < 48.0):
            legal_go = 0.0
        if (yield_req > 0.5 and not car.is_emergency):
            from src.simulation.vehicle import ring_has_priority_traffic
            if ring_has_priority_traffic(car, all_vehicles):
                legal_go = 0.0

        self_stalled = min(1.0, getattr(car, 'time_stalled', 0.0) / 4.0)
        total_len = max(1.0, getattr(car.route, 'total_length', 1.0))
        progress = min(1.0, max(0.0, car.path_distance / total_len))
        queue_go = 1.0 if (
            car.speed < 0.45 and legal_go > 0.5
            and (path_clear > 0.5 or (lead_stalled > 0.5 and (lead_dist or 99) > 12.0)
                 or (lead_dist is None or lead_dist > 14.0))
        ) else 0.0

        dest_code = {'N': 0.0, 'E': 0.33, 'S': 0.66, 'W': 1.0}.get(
            getattr(car.route, 'end_dir', 'S'), 0.5
        )
        # Blend path-tangent fit with intended exit so the policy can tell
        # N→E (south arc) from N→S (east through) on the same approach.
        exit_intent = 0.5 * heading_alignment(car) + 0.5 * dest_code

        return [
            ctx_signal, ctx_ring, ctx_entry, ctx_exit, ctx_follow,
            self_ems, lead_stalled, lead_crossing, path_clear, legal_go,
            self_stalled, progress, queue_go, exit_intent,
        ]


def heading_alignment(car):
    """How well the pose heading matches the route tangent at this arc length."""
    pose = car.route.get_pose_at_distance(car.path_distance)
    if pose is None:
        return 1.0
    _, _, tangent = pose
    gap = abs((car.angle - tangent + math.pi) % (2 * math.pi) - math.pi)
    return max(0.0, 1.0 - gap / math.pi)
