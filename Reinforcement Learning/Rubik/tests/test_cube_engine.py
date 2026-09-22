"""
Astra-DeepCube: Comprehensive Automated Verification Test Suite
Tests Cube Permutations, Whole-Cube Rotations, PyTorch Batched DeepCube Network, ADI, Vision, and GIF Exporter.
"""

import sys
import os
import unittest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.cube_state import CubeState, FACE_U, FACE_D, FACE_F, FACE_B, FACE_L, FACE_R
from core.scrambler import generate_scramble, invert_move, invert_moves
from core.kociemba_solver import KociembaSolver
from core.large_cube_solver import LargeCubeReductionSolver

from rl_engine.deepcube_model import DeepCubeNetwork, TORCH_AVAILABLE, create_default_model
from rl_engine.autodidactic_iteration import AutodidacticTrainer
from rl_engine.neural_astar_search import NeuralAStarSolver

from vision_engine.cube_vision_tracker import CubeVisionTracker
from vision_engine.virtual_multicam_sensor import VirtualMultiCamSensor
from exporter.media_exporter import MediaExporter


class TestCubeEngine(unittest.TestCase):

    def test_01_cube_state_initialization(self):
        cube = CubeState(3)
        self.assertTrue(cube.is_solved())
        self.assertEqual(cube.get_solved_fraction(), 1.0)
        
        c4 = CubeState(4)
        self.assertTrue(c4.is_solved())
        valid, msg = c4.validate_state()
        self.assertTrue(valid)

    def test_02_cube_moves_and_inverses(self):
        cube = CubeState(3)
        cube.apply_moves("R U R' U'")
        self.assertFalse(cube.is_solved())

        cube.apply_moves("U R U' R'")
        self.assertTrue(cube.is_solved())

        for _ in range(6):
            cube.apply_moves("R U R' U'")
        self.assertTrue(cube.is_solved())

    def test_03_whole_cube_and_wide_moves(self):
        cube = CubeState(4)
        cube.apply_move("2Uw")
        cube.apply_move("2Uw'")
        self.assertTrue(cube.is_solved())

        cube.apply_moves("x y z x' y' z'")
        self.assertTrue(cube.is_solved())

    def test_04_tensor_encoding(self):
        cube = CubeState(3)
        one_hot = cube.to_one_hot_tensor()
        self.assertEqual(len(one_hot), 324)
        self.assertEqual(np.sum(one_hot), 54.0)

    def test_05_scrambler(self):
        scramble = generate_scramble(length=15, size=3)
        moves = scramble.split()
        self.assertEqual(len(moves), 15)

    def test_06_kociemba_solver(self):
        solver = KociembaSolver()
        cube = CubeState(3)
        scramble = "R U R' U' F' U F"
        cube.apply_moves(scramble)
        self.assertFalse(cube.is_solved())

        solution = solver.solve(cube)
        self.assertTrue(len(solution) > 0)
        
        cube.apply_moves(solution)
        self.assertTrue(cube.is_solved())

    def test_07_large_cube_solver(self):
        large_solver = LargeCubeReductionSolver()
        c4 = CubeState(4)
        c4.apply_moves("Rw U Rw' U'")
        solution = large_solver.solve(c4)
        self.assertIsInstance(solution, list)

    def test_08_deepcube_network_and_batched_inference(self):
        model = create_default_model("cpu")
        cube = CubeState(3)
        h_val, q_probs = model.predict_single(cube.to_one_hot_tensor())
        self.assertIsInstance(h_val, float)
        self.assertEqual(len(q_probs), 12)

        # Batched inference
        batch_tensors = [cube.to_one_hot_tensor() for _ in range(5)]
        h_vals, q_list = model.predict_batch(batch_tensors)
        self.assertEqual(len(h_vals), 5)
        self.assertEqual(len(q_list), 5)

    def test_09_autodidactic_trainer(self):
        model = create_default_model("cpu")
        trainer = AutodidacticTrainer(model, lr=1e-3, device="cpu")
        metrics = trainer.train_step(batch_size=8, max_scramble_depth=4)
        self.assertIn("loss", metrics)
        self.assertIn("val_loss", metrics)
        self.assertIn("iteration", metrics)
        self.assertEqual(metrics["iteration"], 1)

    def test_10_neural_astar_solver(self):
        model = create_default_model("cpu")
        neural_solver = NeuralAStarSolver(model, weight_lambda=1.2)
        cube = CubeState(3)
        cube.apply_moves("R U R'")
        sol, stats = neural_solver.solve(cube, max_nodes=500, timeout_sec=2.0)
        self.assertIn("nodes_expanded", stats)

    def test_11_vision_multicam(self):
        sensor = VirtualMultiCamSensor(cell_render_size=60)
        cube = CubeState(3)
        matrix_bgr = sensor.render_fly_eye_matrix(cube)
        self.assertEqual(len(matrix_bgr.shape), 3)
        self.assertEqual(matrix_bgr.shape[2], 3)

    def test_12_media_exporter_and_gif(self):
        exporter = MediaExporter(output_dir="tests/test_exports")
        cube = CubeState(3)
        path = exporter.export_solution_card(
            cube_size=3,
            scramble_str="R U R' U'",
            solution_moves=["U", "R", "U'", "R'"],
            solve_time_sec=0.015,
            nodes_expanded=42,
            solver_type="Astra Neural Engine"
        )
        self.assertTrue(os.path.exists(path))

        gif_path = exporter.export_solution_gif(
            cube_initial=cube,
            solution_moves=["U", "R", "U'", "R'"]
        )
        self.assertTrue(os.path.exists(gif_path))


if __name__ == "__main__":
    unittest.main()
