import os
import sys
import unittest

os.environ["SDL_VIDEODRIVER"] = "dummy"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math

from src.config import NUM_LANES_PER_DIR, VISION_STATE_SIZE, STACKED_STATE_SIZE
from src.simulation.intersection import Intersection
from src.simulation.vehicle import Vehicle
from src.simulation.traffic_controller import TrafficController


class TestCityNetwork(unittest.TestCase):
    def setUp(self):
        self.city = Intersection(width=1600, height=900, topnav_h=48, sidebar_w=340)

    def test_dual_nodes_and_lanes(self):
        self.assertEqual(NUM_LANES_PER_DIR, 3)
        self.assertGreater(self.city.rby, self.city.cy + self.city.half_rw)
        self.assertGreater(len(self.city.routes), 12)
        nodes = {r.node for r in self.city.routes}
        self.assertIn('MAIN', nodes)
        self.assertIn('ROUNDABOUT', nodes)
        self.assertIn('THROUGH', nodes)

    def test_routes_have_length(self):
        for route in self.city.routes:
            self.assertGreater(route.total_length, 20.0, msg=f"route {route.id}")
            pose = route.get_pose_at_distance(0.0)
            self.assertIsNotNone(pose)

    def test_roundabout_conflict_bounds(self):
        rb = next(r for r in self.city.routes if r.node == 'ROUNDABOUT')
        bounds = self.city.conflict_bounds_for(rb)
        self.assertEqual(bounds, self.city.roundabout_bounds)

    def test_yield_light_on_roundabout(self):
        ctl = TrafficController()
        self.assertEqual(ctl.get_light_state('W', node='ROUNDABOUT'), 'YIELD')
        self.assertIn(ctl.get_light_state('N'), ('GREEN', 'YELLOW', 'RED'))

    def test_emergency_profiles_and_state_dim(self):
        route = self.city.routes[0]
        fire = Vehicle(route, v_type='FIRE', spawn_speed=2.0)
        police = Vehicle(route, v_type='POLICE', spawn_speed=2.0)
        self.assertTrue(fire.is_emergency)
        self.assertTrue(police.is_emergency)
        raw = fire.sensors.update([fire], 'GREEN', self.city.junction_bounds)
        self.assertEqual(raw.shape[0], VISION_STATE_SIZE)
        self.assertEqual(VISION_STATE_SIZE * 3, STACKED_STATE_SIZE)

    def _main_connector_route(self, start, end):
        matches = [
            r for r in self.city.routes
            if r.start_dir == start and r.end_dir == end and r.node != 'ROUNDABOUT'
        ]
        self.assertTrue(matches, msg=f"missing connector {start}->{end}")
        return matches[0]

    def test_ws_es_continue_through_roundabout(self):
        """W->S and E->S from the 4-way must enter the circle and exit south."""
        for start, end in (('W', 'S'), ('E', 'S')):
            route = self._main_connector_route(start, end)
            self.assertEqual(route.node, 'THROUGH')
            self.assertIsNotNone(route.yield_line_dist)
            min_d = min(
                math.hypot(x - self.city.rbx, y - self.city.rby)
                for x, y, *_ in route.samples
            )
            self.assertLess(min_d, self.city.r_outer - 8.0, msg=f"{start}->{end} never enters ring")
            end_x, end_y, _ = route.samples[-1]
            self.assertGreater(end_y, self.city.rby + self.city.r_outer, msg=f"{start}->{end} does not exit south")
            mouth_y = self.city.rby - self.city.r_outer - 20
            self.assertGreater(end_y, mouth_y + 40.0, msg=f"{start}->{end} still dead-ends at north mouth")

    def test_civilian_keeps_moving_in_ring_when_ems_is_behind(self):
        route = next(r for r in self.city.routes if r.node == 'ROUNDABOUT' and r.start_dir == 'W' and r.end_dir == 'E')
        civilian = Vehicle(route, v_type='SEDAN', spawn_speed=3.0)
        ems = Vehicle(route, v_type='AMBULANCE', spawn_speed=5.0)
        ring_d = None
        for dist, (x, y, *_) in zip(route.cumulative_dist, route.samples):
            if math.hypot(x - self.city.rbx, y - self.city.rby) < self.city.r_outer - 10.0:
                ring_d = dist
                break
        self.assertIsNotNone(ring_d)
        civilian.path_distance = ring_d
        ems.path_distance = max(0.0, civilian.path_distance - 55.0)
        civilian.update_pose_from_path()
        ems.update_pose_from_path()
        self.assertLess(
            math.hypot(civilian.x - self.city.rbx, civilian.y - self.city.rby),
            self.city.r_outer + 4.0,
        )
        for _ in range(60):
            civilian.apply_action(2)
            ems.apply_action(2)
            civilian.update_physics(dt=1.0 / 60.0, current_tl_state='YIELD', all_vehicles=[civilian, ems],
                                    conflict_bounds=self.city.roundabout_bounds)
            ems.update_physics(dt=1.0 / 60.0, current_tl_state='YIELD', all_vehicles=[civilian, ems],
                               conflict_bounds=self.city.roundabout_bounds)
            in_ring = math.hypot(civilian.x - self.city.rbx, civilian.y - self.city.rby) < self.city.r_outer + 8.0
            if in_ring:
                self.assertGreater(civilian.speed, 1.0, msg="civilian parked in the ring in front of EMS")
                self.assertGreater(ems.speed, 1.2, msg="EMS boxed in by a stopped civilian")
        self.assertGreater(civilian.pull_over_offset, 0.0)

    def test_roundabout_has_two_circulating_radii(self):
        self.assertGreater(self.city.r_circ - self.city.r_inner, 20.0)
        self.assertGreater(self.city.r_inner, self.city.r_island)

    def test_north_entries_stay_off_the_yellow(self):
        """Southbound traffic must first meet the ring west of the NS centerline."""
        rbx, rby = self.city.rbx, self.city.rby
        for route in self.city.routes:
            if route.start_dir != 'N':
                continue
            if getattr(route, 'node', 'MAIN') not in ('THROUGH', 'ROUNDABOUT'):
                continue
            enter_x = None
            for x, y, *_ in route.samples:
                dist = math.hypot(x - rbx, y - rby)
                if dist <= self.city.r_outer - 4.0 and y < rby - 10.0:
                    enter_x = x
                    break
            self.assertIsNotNone(
                enter_x,
                msg=f"{route.start_dir}->{route.end_dir} L{route.lane_index} never meets the north mouth",
            )
            self.assertLess(
                enter_x, rbx - 6.0,
                msg=f"{route.start_dir}->{route.end_dir} L{route.lane_index} joins on the yellow",
            )

    def test_boulevard_uses_three_lanes(self):
        for start, end in (('W', 'E'), ('E', 'W'), ('W', 'S'), ('S', 'E')):
            lanes = [r.lane_index for r in self.city.routes
                     if r.node == 'ROUNDABOUT' and r.start_dir == start and r.end_dir == end]
            self.assertEqual(sorted(set(lanes)), [0, 1, 2], msg=f"{start}->{end}")

    def test_all_ring_routes_circulate_ccw(self):
        """No route may drive the circle clockwise / against the flow."""
        rbx, rby = self.city.rbx, self.city.rby
        for route in self.city.routes:
            if getattr(route, 'node', 'MAIN') not in ('ROUNDABOUT', 'THROUGH'):
                continue
            self.assertTrue(getattr(route, 'circ_ccw', True), msg=f"route {route.id} marked CW")
            thetas = []
            for x, y, *_ in route.samples:
                dist = math.hypot(x - rbx, y - rby)
                on_lane = (
                    abs(dist - self.city.r_circ) < 10.0
                    or abs(dist - self.city.r_inner) < 10.0
                )
                if on_lane:
                    thetas.append(math.atan2(rby - y, x - rbx))
            if len(thetas) < 6:
                continue
            pos = 0
            neg = 0
            for a, b in zip(thetas, thetas[1:]):
                step = (b - a + math.pi) % (2.0 * math.pi) - math.pi
                if step > math.radians(1.0):
                    pos += 1
                elif step < -math.radians(1.0):
                    neg += 1
            if pos + neg < 3:
                continue
            self.assertGreater(
                pos, neg,
                msg=f"{route.start_dir}->{route.end_dir} L{route.lane_index} goes the wrong way on the ring",
            )

    def test_roundabout_exits_stay_on_travel_lanes(self):
        rbx, rby = self.city.rbx, self.city.rby
        ne = next(r for r in self.city.routes
                  if r.node == 'THROUGH' and r.start_dir == 'N' and r.end_dir == 'E')
        _, ey, _ = ne.samples[-1]
        self.assertGreater(ey, rby + 6.0, msg="N->E must exit on eastbound (south) lanes")

        es = next(r for r in self.city.routes
                  if r.node == 'ROUNDABOUT' and r.start_dir == 'E' and r.end_dir == 'S')
        sx, _, _ = es.samples[-1]
        self.assertLess(sx, rbx - 6.0, msg="E->S must exit on southbound (west) lanes")

        ns = next(r for r in self.city.routes
                  if r.node == 'THROUGH' and r.start_dir == 'N' and r.end_dir == 'S')
        ns_ring = [
            (x, y) for x, y, *_ in ns.samples
            if self.city.r_island + 6.0 < math.hypot(x - rbx, y - rby) < self.city.r_outer
        ]
        self.assertGreater(
            sum(1 for x, _ in ns_ring if x < rbx - 8.0),
            sum(1 for x, _ in ns_ring if x > rbx + 8.0),
            msg="CCW N->S through must use the west side of the island",
        )


if __name__ == '__main__':
    unittest.main()
