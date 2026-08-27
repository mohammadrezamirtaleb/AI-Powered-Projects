import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulation.vehicle import check_sat_collision, get_rotated_rect_corners

class TestVehicleCollision(unittest.TestCase):
    def test_sat_collision_overlapping(self):
        corners_a = get_rotated_rect_corners(0, 0, 10, 10, 0)
        corners_b = get_rotated_rect_corners(5, 5, 10, 10, 0)
        self.assertTrue(check_sat_collision(corners_a, corners_b))

    def test_sat_collision_separated(self):
        corners_a = get_rotated_rect_corners(0, 0, 10, 10, 0)
        corners_b = get_rotated_rect_corners(20, 20, 10, 10, 0)
        self.assertFalse(check_sat_collision(corners_a, corners_b))

    def test_sat_collision_rotated_overlap(self):
        corners_a = get_rotated_rect_corners(0, 0, 10, 2, 0)
        # 90 degrees rotated, overlapping at origin
        corners_b = get_rotated_rect_corners(0, 0, 2, 10, 1.57079632679)
        self.assertTrue(check_sat_collision(corners_a, corners_b))

if __name__ == '__main__':
    unittest.main()
