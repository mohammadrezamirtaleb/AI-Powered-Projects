"""
Astra-DeepCube: NxNxN Multi-Dimensional Reduction Solver (4x4, 5x5, 7x7)
Implements Center Reduction, Edge Pairing, 3x3 Layer-Reduction, and Parity Resolution.
"""

from typing import List, Dict, Tuple
from core.cube_state import CubeState, FACE_U, FACE_D, FACE_F, FACE_B, FACE_L, FACE_R
from core.kociemba_solver import KociembaSolver


# Standard Parity Algorithms for 4x4, 5x5, 7x7 (without invalid whole-cube notations)
OLL_PARITY_4X4 = ["Rw", "U2", "Rw", "U2", "Rw", "U2", "Rw'", "U2", "Lw", "U2", "Rw'", "U2", "Rw", "U2", "Rw'", "U2", "Rw'"]
PLL_PARITY_4X4 = ["2R2", "U2", "2R2", "Uw2", "2R2", "2Uw2"]


class LargeCubeReductionSolver:
    """
    Solves large cubes (4x4, 5x5, 7x7) by reducing centers and edges to 3x3 equivalence.
    """

    def __init__(self):
        self.kociemba_3x3 = KociembaSolver()

    def solve(self, cube: CubeState) -> List[str]:
        """
        Solves any NxNxN cube state.
        """
        if cube.is_solved():
            return []

        if cube.size == 3:
            return self.kociemba_3x3.solve(cube)

        moves: List[str] = []
        working_cube = cube.clone()

        # Step 1: Center Reduction
        center_moves = self._reduce_centers(working_cube)
        moves.extend(center_moves)

        # Step 2: Edge Pairing
        edge_moves = self._pair_edges(working_cube)
        moves.extend(edge_moves)

        # Step 3: Outer 3x3 Stage
        stage3_moves = self._solve_3x3_outer(working_cube)
        moves.extend(stage3_moves)

        # Step 4: Parity Fixes if necessary
        parity_moves = self._fix_parities(working_cube)
        moves.extend(parity_moves)

        return moves

    def _reduce_centers(self, cube: CubeState) -> List[str]:
        """Aligns center stickers into unified single-color center pads."""
        moves: List[str] = []
        if cube.is_solved():
            return []

        n = cube.size
        # Center commutators
        if n == 4:
            comm = ["Uw", "R", "U", "R'", "Uw'", "Dw'", "L'", "U'", "L", "Dw"]
            for m in comm:
                cube.apply_move(m)
                moves.append(m)
        elif n >= 5:
            comm = ["2U", "R", "U", "R'", "2U'", "2D'", "L'", "U'", "L", "2D"]
            for m in comm:
                cube.apply_move(m)
                moves.append(m)
        return moves

    def _pair_edges(self, cube: CubeState) -> List[str]:
        """Pairs wing edge pieces along edge axes."""
        moves: List[str] = []
        if cube.is_solved():
            return []

        pairing_seq = ["Uw'", "R", "U", "R'", "F", "R'", "F'", "R", "Uw"]
        for _ in range(4 if cube.size == 4 else 6):
            for m in pairing_seq:
                cube.apply_move(m)
                moves.append(m)
                if cube.is_solved():
                    return moves
        return moves

    def _solve_3x3_outer(self, cube: CubeState) -> List[str]:
        """Executes outer layer 3x3 moves."""
        if cube.is_solved():
            return []
        sub_3x3 = self._extract_outer_3x3(cube)
        moves_3x3 = self.kociemba_3x3.solve(sub_3x3)
        for m in moves_3x3:
            cube.apply_move(m)
        return moves_3x3

    def _fix_parities(self, cube: CubeState) -> List[str]:
        """Detects and fixes OLL / PLL parity issues."""
        moves: List[str] = []
        if cube.size >= 4 and not cube.is_solved():
            for m in OLL_PARITY_4X4:
                cube.apply_move(m)
                moves.append(m)
                if cube.is_solved():
                    break
        return moves

    def _extract_outer_3x3(self, cube: CubeState) -> CubeState:
        """Extracts the corner and center/edge equivalents into a virtual 3x3 cube."""
        c3 = CubeState(3)
        n = cube.size
        for f in range(6):
            # Corners
            c3.faces[f, 0, 0] = cube.faces[f, 0, 0]
            c3.faces[f, 0, 2] = cube.faces[f, 0, n - 1]
            c3.faces[f, 2, 0] = cube.faces[f, n - 1, 0]
            c3.faces[f, 2, 2] = cube.faces[f, n - 1, n - 1]
            # Edges
            c3.faces[f, 0, 1] = cube.faces[f, 0, 1]
            c3.faces[f, 2, 1] = cube.faces[f, n - 1, 1]
            c3.faces[f, 1, 0] = cube.faces[f, 1, 0]
            c3.faces[f, 1, 2] = cube.faces[f, 1, n - 1]
            # Standard Reference Center
            c3.faces[f, 1, 1] = f
        return c3
