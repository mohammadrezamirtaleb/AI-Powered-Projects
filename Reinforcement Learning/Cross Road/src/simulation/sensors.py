"""
Perception and Computer Vision / LiDAR Sensing System for Autonomous Vehicles.
Performs 11-ray directional raycasting, Time-to-Collision (TTC) estimation,
traffic light detection, relative speed estimation, and conflict zone radar.
"""
import math
import numpy as np
from src.config import LIDAR_NUM_RAYS, LIDAR_MAX_DIST, LIDAR_FOV

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
        is_before_stop = dist_to_stop is not None and dist_to_stop >= -15.0

        if is_before_stop:
            tl_state = traffic_light_state
            norm_stop_dist = max(0.0, min(1.0, dist_to_stop / 250.0)) if dist_to_stop > 0 else 0.0
        else:
            tl_state = 'NONE'
            norm_stop_dist = 0.0

        tl_one_hot = [
            1.0 if tl_state == 'RED' else 0.0,
            1.0 if tl_state == 'YELLOW' else 0.0,
            1.0 if tl_state == 'GREEN' else 0.0,
            1.0 if tl_state == 'NONE' else 0.0,
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

        # Assemble full 29-dimensional state vector
        state = np.array(
            ray_distances +                # 9
            ray_rel_speeds +               # 9
            tl_one_hot +                   # 4
            [norm_stop_dist] +             # 1
            [norm_speed] +                 # 1
            [norm_target_speed] +          # 1
            [norm_lead_dist] +             # 1
            [norm_junction_density, crossing_hazard] + # 2
            [friction_coeff],              # 1
            dtype=np.float32
        )
        return state
