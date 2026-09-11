import os
import sys
import math
import unittest

os.environ["SDL_VIDEODRIVER"] = "dummy"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch

from src.config import VISION_STATE_SIZE, STACKED_STATE_SIZE, SITUATION_FEATURE_SIZE
from src.simulation.intersection import Intersection
from src.simulation.vehicle import Vehicle
from src.ai.network import DuelingDQN
from src.train_headless import get_expert_action, apply_safety_shield


class TestSituationPolicy(unittest.TestCase):
    def setUp(self):
        self.city = Intersection(width=1600, height=900, topnav_h=48, sidebar_w=340)

    def test_observation_includes_situation_slice(self):
        route = self.city.routes[0]
        car = Vehicle(route, v_type='SEDAN', spawn_speed=2.0)
        raw = car.sensors.update([car], 'GREEN', self.city.junction_bounds)
        self.assertEqual(raw.shape[0], VISION_STATE_SIZE)
        self.assertEqual(VISION_STATE_SIZE, 38 + SITUATION_FEATURE_SIZE)
        self.assertEqual(STACKED_STATE_SIZE, VISION_STATE_SIZE * 3)

    def test_situation_net_forward_matches_stack_dim(self):
        net = DuelingDQN()
        x = torch.zeros(2, STACKED_STATE_SIZE)
        q = net(x)
        self.assertEqual(tuple(q.shape), (2, 5))
        acts = net.get_layer_activations(x[0])
        self.assertEqual(len(acts['q_values']), 5)

    def test_go_shield_unsticks_a_clear_stopped_car(self):
        route = next(r for r in self.city.routes if r.node == 'ROUNDABOUT')
        car = Vehicle(route, v_type='SEDAN', spawn_speed=0.0)
        car.path_distance = route.total_length * 0.45
        car.has_cleared_yield = True
        car.update_pose_from_path()
        car.sensors.update([car], 'YIELD', self.city.roundabout_bounds)
        action = apply_safety_shield(car, 0, 'YIELD', [car], self.city.roundabout_bounds)
        self.assertIn(action, (1, 2))

    def test_expert_accelerates_in_an_empty_ring(self):
        route = next(r for r in self.city.routes if r.node == 'ROUNDABOUT')
        car = Vehicle(route, v_type='SEDAN', spawn_speed=0.5)
        car.path_distance = route.total_length * 0.45
        car.has_cleared_yield = True
        car.update_pose_from_path()
        car.sensors.update([car], 'YIELD', self.city.roundabout_bounds)
        action = get_expert_action(car, 'YIELD', car.get_distance_to_stop_line(), [car], self.city.roundabout_bounds)
        self.assertIn(action, (1, 2))

    def test_ring_flow_prevents_zero_speed_inside_circle(self):
        route = next(
            r for r in self.city.routes
            if r.node == 'ROUNDABOUT' and r.start_dir == 'W' and r.end_dir == 'E'
        )
        car = Vehicle(route, v_type='SEDAN', spawn_speed=0.0)
        placed = False
        for dist, (x, y, *_) in zip(route.cumulative_dist, route.samples):
            if math.hypot(x - self.city.rbx, y - self.city.rby) < self.city.r_outer - 10.0:
                car.path_distance = dist
                placed = True
                break
        self.assertTrue(placed)
        car.has_cleared_yield = True
        car.update_pose_from_path()
        car.apply_action(0)
        car.update_physics(dt=1.0 / 60.0, current_tl_state='YIELD', all_vehicles=[car],
                           conflict_bounds=self.city.roundabout_bounds)
        self.assertGreater(car.speed, 0.9)
        d = math.hypot(car.x - self.city.rbx, car.y - self.city.rby)
        self.assertLess(d, self.city.r_outer)


if __name__ == '__main__':
    unittest.main()
