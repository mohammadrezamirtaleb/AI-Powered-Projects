"""
Reward and penalty function tests.

The point of these is the balance between terms, not the exact numbers: a policy
must never find crashing cheaper than driving slowly, and a violation must always
cost more than the time it saves.
"""
import os
import sys
import math
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.ai.dqn_agent import DQNAgent
from src.simulation.intersection import Intersection
from src.simulation.vehicle import Vehicle, attribute_fault

DT = 1.0 / 60.0


def make_car(intersection, node=None, turn=None):
    for route in intersection.routes:
        if node is not None and getattr(route, 'node', 'MAIN') != node:
            continue
        if turn is not None and route.turn_type != turn:
            continue
        return Vehicle(route, v_type='SEDAN', spawn_speed=3.0)
    raise AssertionError(f"no route matching node={node} turn={turn}")


class TestRewardBalance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = DQNAgent()
        cls.city = Intersection()

    def test_dense_terms_are_per_second(self):
        """Doubling dt must double a dense term, so the frame rate cannot shift the balance."""
        car = make_car(self.city)
        car.speed = 3.0
        car.target_speed = 5.0
        r1 = self.agent.calculate_reward(car, 'GREEN', 500.0, dt=DT)
        car2 = make_car(self.city)
        car2.speed = 3.0
        car2.target_speed = 5.0
        r2 = self.agent.calculate_reward(car2, 'GREEN', 500.0, dt=DT * 2)
        self.assertAlmostEqual(r2, r1 * 2, places=5)

    def test_crash_outweighs_a_whole_clean_trip(self):
        """
        A full lawful crossing must be worth less than an at-fault crash costs,
        otherwise the policy can pay for collisions with progress reward.
        """
        car = make_car(self.city)
        car.speed = car.target_speed
        # 15 seconds of best-case dense progress
        dense = 0.0
        for _ in range(15 * 60):
            dense += self.agent.calculate_reward(car, 'GREEN', 500.0, dt=DT)
        best_trip = dense + config.REWARD_PASS_EVENT + config.REWARD_ROUTE_COMPLETE
        self.assertLess(best_trip, abs(config.PENALTY_CRASH),
                        f"a clean trip is worth {best_trip:.1f}, crash costs "
                        f"{abs(config.PENALTY_CRASH):.1f}")

    def test_running_a_red_costs_more_than_waiting(self):
        """Waiting a full red phase must be cheaper than crossing it."""
        car = make_car(self.city)
        car.speed = 0.0
        waiting = 0.0
        red_seconds = (config.PHASE_DURATIONS['EW_GREEN'] + config.PHASE_DURATIONS['EW_YELLOW']
                       + config.PHASE_DURATIONS['ALL_RED_1'] + config.PHASE_DURATIONS['ALL_RED_2'])
        for _ in range(int(red_seconds * 60)):
            waiting += self.agent.calculate_reward(car, 'RED', 20.0, dt=DT)
        self.assertGreater(waiting, config.PENALTY_RED_LIGHT_RUN)

    def test_pedestrian_hit_is_the_worst_outcome(self):
        self.assertLess(config.PENALTY_CRASH_PEDESTRIAN, config.PENALTY_CRASH)

    def test_at_fault_crash_is_penalized_and_innocent_is_not(self):
        guilty = make_car(self.city)
        guilty.crash(is_at_fault=True)
        innocent = make_car(self.city)
        innocent.crash(is_at_fault=False)
        self.assertEqual(self.agent.calculate_reward(guilty, 'GREEN', 500.0, dt=DT),
                         config.PENALTY_CRASH)
        self.assertEqual(self.agent.calculate_reward(innocent, 'GREEN', 500.0, dt=DT), 0.0)

    def test_pedestrian_collision_uses_the_pedestrian_weight(self):
        car = make_car(self.city)
        car.crash(is_at_fault=True, hit_pedestrian=True)
        self.assertEqual(self.agent.calculate_reward(car, 'GREEN', 500.0, dt=DT),
                         config.PENALTY_CRASH_PEDESTRIAN)

    def test_roundabout_route_can_earn_a_terminal_reward(self):
        """
        Roundabout routes carry a sentinel stop line of 1e6, so they can never
        set has_passed_intersection. Route completion is their only success signal.
        """
        car = make_car(self.city, node='ROUNDABOUT')
        self.assertGreater(car.route.stop_line_dist, 4000)
        car.has_finished = True
        car.speed = 0.0
        reward = self.agent.calculate_reward(car, 'YIELD', car.get_distance_to_stop_line(), dt=DT)
        self.assertGreater(reward, config.REWARD_ROUTE_COMPLETE * 0.9)

    def test_terminal_reward_fires_only_once(self):
        car = make_car(self.city, node='ROUNDABOUT')
        car.has_finished = True
        first = self.agent.calculate_reward(car, 'YIELD', None, dt=DT)
        second = self.agent.calculate_reward(car, 'YIELD', None, dt=DT)
        self.assertGreater(first, config.REWARD_ROUTE_COMPLETE * 0.9)
        self.assertLess(second, 1.0)

    def test_yield_violation_is_penalized_once(self):
        car = make_car(self.city, node='ROUNDABOUT')
        car.speed = 2.0
        car.yield_violated = True
        first, parts = self.agent.calculate_penalty(car, 'YIELD', None, dt=DT)
        self.assertIn('yield_violation', parts)
        _, parts2 = self.agent.calculate_penalty(car, 'YIELD', None, dt=DT)
        self.assertNotIn('yield_violation', parts2)

    def test_tailgating_scales_with_severity(self):
        near = make_car(self.city)
        near.speed = 4.0
        near.last_lead_clearance = 5.0
        far = make_car(self.city)
        far.speed = 4.0
        far.last_lead_clearance = 30.0
        _, near_parts = self.agent.calculate_penalty(near, 'GREEN', 500.0, dt=DT)
        _, far_parts = self.agent.calculate_penalty(far, 'GREEN', 500.0, dt=DT)
        self.assertLess(near_parts['tailgate'], far_parts['tailgate'])

    def test_speeding_is_penalized(self):
        car = make_car(self.city)
        car.speed = car.target_speed * 1.5
        _, parts = self.agent.calculate_penalty(car, 'GREEN', 500.0, dt=DT)
        self.assertIn('speeding', parts)

    def test_blocking_the_conflict_box_is_penalized(self):
        car = make_car(self.city)
        car.speed = 0.0
        car.time_stalled = 2.0
        car.in_conflict_box = True
        _, parts = self.agent.calculate_penalty(car, 'GREEN', 500.0, dt=DT)
        self.assertIn('blocking_box', parts)

    def test_waiting_at_red_is_not_treated_as_a_stall(self):
        car = make_car(self.city)
        car.speed = 0.0
        car.time_stalled = 5.0
        car.in_conflict_box = True
        _, parts = self.agent.calculate_penalty(car, 'RED', 20.0, dt=DT)
        self.assertNotIn('stall', parts)
        self.assertNotIn('blocking_box', parts)
        self.assertNotIn('time', parts)


class TestFaultAttribution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.city = Intersection()

    def test_rear_end_blames_the_follower(self):
        follower = make_car(self.city)
        leader = make_car(self.city)
        follower.x, follower.y, follower.angle = 0.0, 0.0, 0.0
        leader.x, leader.y, leader.angle = 40.0, 0.0, 0.0
        follower.speed = 3.0
        leader.speed = 3.0
        f_fault, l_fault = attribute_fault(follower, leader)
        self.assertTrue(f_fault)
        self.assertFalse(l_fault)

    def test_crossing_blames_the_red_light_runner(self):
        legal = make_car(self.city)
        runner = make_car(self.city)
        legal.x, legal.y, legal.angle = 0.0, 0.0, 0.0
        runner.x, runner.y, runner.angle = 20.0, 10.0, math.pi / 2
        legal.speed = 3.0
        runner.speed = 3.0
        legal.tl_state_at_entry = 'GREEN'
        runner.tl_state_at_entry = 'RED'
        legal_fault, runner_fault = attribute_fault(legal, runner)
        self.assertFalse(legal_fault)
        self.assertTrue(runner_fault)

    def test_moving_into_a_stationary_vehicle_blames_the_mover(self):
        mover = make_car(self.city)
        parked = make_car(self.city)
        mover.x, mover.y, mover.angle = 0.0, 0.0, 0.0
        parked.x, parked.y, parked.angle = 15.0, 12.0, math.pi / 2
        mover.speed = 3.0
        parked.speed = 0.0
        m_fault, p_fault = attribute_fault(mover, parked)
        self.assertTrue(m_fault)
        self.assertFalse(p_fault)


if __name__ == '__main__':
    unittest.main(verbosity=2)
