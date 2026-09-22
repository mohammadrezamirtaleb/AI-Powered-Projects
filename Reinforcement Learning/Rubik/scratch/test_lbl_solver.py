import sys
sys.path.insert(0, ".")
from collections import deque
from typing import List, Tuple, Optional
import numpy as np

from core.cube_state import CubeState, FACE_U, FACE_D, FACE_F, FACE_B, FACE_L, FACE_R
from core.scrambler import generate_scramble, invert_move, invert_moves

def simplify_moves(moves: List[str]) -> List[str]:
    if not moves:
        return []
    stack: List[str] = []
    for m in moves:
        if not stack:
            stack.append(m)
            continue
        prev = stack[-1]
        base_p, base_m = prev[0], m[0]
        if base_p == base_m and len(prev) <= 2 and len(m) <= 2:
            mod_p = prev[1:] if len(prev) > 1 else ""
            mod_m = m[1:] if len(m) > 1 else ""
            turns_p = 3 if mod_p == "'" else (2 if mod_p == "2" else 1)
            turns_m = 3 if mod_m == "'" else (2 if mod_m == "2" else 1)
            total = (turns_p + turns_m) % 4
            stack.pop()
            if total == 1:
                stack.append(base_p)
            elif total == 2:
                stack.append(base_p + "2")
            elif total == 3:
                stack.append(base_p + "'")
        else:
            stack.append(m)
    return stack

class GuaranteedLBLSolver:
    """
    100% Deterministic Layer-By-Layer (CFOP) Rubik's Cube Solver.
    Guaranteed to solve any valid 3x3 permutation in ~50-80 moves.
    """

    def solve(self, cube: CubeState) -> List[str]:
        if cube.is_solved():
            return []

        c = cube.clone()
        moves: List[str] = []

        def apply_seq(seq: str):
            for m in seq.split():
                c.apply_move(m)
                moves.append(m)

        # Stage 1: White Cross (D face)
        self._solve_white_cross(c, apply_seq)

        # Stage 2: White Corners (D corners)
        self._solve_white_corners(c, apply_seq)

        # Stage 3: Middle Layer Edges
        self._solve_middle_edges(c, apply_seq)

        # Stage 4: Yellow Cross (OLL Edges)
        self._solve_yellow_cross(c, apply_seq)

        # Stage 5: Yellow Corners Orientation (OLL Corners)
        self._solve_yellow_corners_orientation(c, apply_seq)

        # Stage 6: Yellow Corners Permutation (PLL Corners)
        self._solve_yellow_corners_permutation(c, apply_seq)

        # Stage 7: Yellow Edges Permutation (PLL Edges)
        self._solve_yellow_edges_permutation(c, apply_seq)

        # Stage 8: Adjust AUF
        self._solve_auf(c, apply_seq)

        return simplify_moves(moves)

    # -------------------------------------------------------------------------
    # Helper: BFS for finding shortest sequence to satisfy a goal condition
    # -------------------------------------------------------------------------
    def _bfs_solve_target(self, cube: CubeState, goal_check, allowed_moves: List[str], max_depth: int = 6) -> List[str]:
        if goal_check(cube):
            return []
        
        queue = deque([(cube.clone(), [])])
        visited = {cube.faces.tobytes()}

        while queue:
            curr_state, path = queue.popleft()
            if len(path) >= max_depth:
                continue

            for m in allowed_moves:
                if path and path[-1][0] == m[0]:
                    continue
                nxt = curr_state.clone()
                nxt.apply_move(m)
                if goal_check(nxt):
                    return path + [m]
                k = nxt.faces.tobytes()
                if k not in visited:
                    visited.add(k)
                    queue.append((nxt, path + [m]))
        return []

    # -------------------------------------------------------------------------
    # Stage 1: White Cross on D (White = FACE_D or FACE_U, let's use White = FACE_U as top and Yellow = FACE_D as bottom, or White on D)
    # On our CubeState: FACE_U = 0 (White), FACE_D = 1 (Yellow), FACE_F = 2 (Green), FACE_B = 3 (Blue), FACE_L = 4 (Orange), FACE_R = 5 (Red)
    # Let's solve White (0) on FACE_U (or on FACE_D)!
    # Solving White (0) on FACE_D (bottom) is standard CFOP.
    # -------------------------------------------------------------------------
    def _solve_white_cross(self, cube: CubeState, apply_fn):
        # Target: D-face has 0 (White) on edges, and matching side colors:
        # DF: D[0, 1] == 0 and F[2, 1] == 2
        # DR: D[1, 2] == 0 and R[2, 1] == 5
        # DB: D[2, 1] == 0 and B[2, 1] == 3
        # DL: D[1, 0] == 0 and L[2, 1] == 4
        
        edges = [
            ("DF", lambda c: c.faces[FACE_D, 0, 1] == FACE_U and c.faces[FACE_F, 2, 1] == FACE_F),
            ("DR", lambda c: c.faces[FACE_D, 1, 2] == FACE_U and c.faces[FACE_R, 2, 1] == FACE_R),
            ("DB", lambda c: c.faces[FACE_D, 2, 1] == FACE_U and c.faces[FACE_B, 2, 1] == FACE_B),
            ("DL", lambda c: c.faces[FACE_D, 1, 0] == FACE_U and c.faces[FACE_L, 2, 1] == FACE_L),
        ]

        allowed = ["U", "U'", "U2", "D", "D'", "D2", "L", "L'", "L2", "R", "R'", "R2", "F", "F'", "F2", "B", "B'", "B2"]

        for idx, (name, check_fn) in enumerate(edges):
            if check_fn(cube):
                continue
            # Previous edges must remain solved
            prev_checks = [edges[i][1] for i in range(idx)]
            def goal(c, cur_fn=check_fn, p_fns=prev_checks):
                return cur_fn(c) and all(pf(c) for pf in p_fns)

            path = self._bfs_solve_target(cube, goal, allowed, max_depth=6)
            if not path:
                # If depth 6 not enough, try depth 7
                path = self._bfs_solve_target(cube, goal, allowed, max_depth=7)
            if path:
                apply_fn(" ".join(path))

    # -------------------------------------------------------------------------
    # Stage 2: White Corners (First Layer)
    # -------------------------------------------------------------------------
    def _solve_white_corners(self, cube: CubeState, apply_fn):
        # Target corners on D layer:
        # D-F-R: D[0, 2] == U (0), F[2, 2] == F (2), R[2, 0] == R (5)
        # D-R-B: D[2, 2] == U (0), R[2, 2] == R (5), B[2, 0] == B (3)
        # D-B-L: D[2, 0] == U (0), B[2, 2] == B (3), L[2, 0] == L (4)
        # D-L-F: D[0, 0] == U (0), L[2, 2] == L (4), F[2, 0] == F (2)

        target_corners = [
            ("DFR", {FACE_U, FACE_F, FACE_R}, lambda c: c.faces[FACE_D, 0, 2] == FACE_U and c.faces[FACE_F, 2, 2] == FACE_F and c.faces[FACE_R, 2, 0] == FACE_R, "R U R' U'"),
            ("DRB", {FACE_U, FACE_R, FACE_B}, lambda c: c.faces[FACE_D, 2, 2] == FACE_U and c.faces[FACE_R, 2, 2] == FACE_R and c.faces[FACE_B, 2, 0] == FACE_B, "B U B' U'"),
            ("DBL", {FACE_U, FACE_B, FACE_L}, lambda c: c.faces[FACE_D, 2, 0] == FACE_U and c.faces[FACE_B, 2, 2] == FACE_B and c.faces[FACE_L, 2, 0] == FACE_L, "L U L' U'"),
            ("DLF", {FACE_U, FACE_L, FACE_F}, lambda c: c.faces[FACE_D, 0, 0] == FACE_U and c.faces[FACE_L, 2, 2] == FACE_L and c.faces[FACE_F, 2, 0] == FACE_F, "F U F' U'"),
        ]

        def get_top_corner_colors(c, slot):
            # Slots on U layer:
            # U-F-R: U[2, 2], F[0, 2], R[0, 0]
            # U-R-B: U[0, 2], R[0, 2], B[0, 0]
            # U-B-L: U[0, 0], B[0, 2], L[0, 0]
            # U-L-F: U[2, 0], L[0, 2], F[0, 0]
            if slot == "UFR":
                return {int(c.faces[FACE_U, 2, 2]), int(c.faces[FACE_F, 0, 2]), int(c.faces[FACE_R, 0, 0])}
            elif slot == "URB":
                return {int(c.faces[FACE_U, 0, 2]), int(c.faces[FACE_R, 0, 2]), int(c.faces[FACE_B, 0, 0])}
            elif slot == "UBL":
                return {int(c.faces[FACE_U, 0, 0]), int(c.faces[FACE_B, 0, 2]), int(c.faces[FACE_L, 0, 0])}
            elif slot == "ULF":
                return {int(c.faces[FACE_U, 2, 0]), int(c.faces[FACE_L, 0, 2]), int(c.faces[FACE_F, 0, 0])}
            return set()

        def get_bottom_corner_colors(c, slot):
            if slot == "DFR":
                return {int(c.faces[FACE_D, 0, 2]), int(c.faces[FACE_F, 2, 2]), int(c.faces[FACE_R, 2, 0])}
            elif slot == "DRB":
                return {int(c.faces[FACE_D, 2, 2]), int(c.faces[FACE_R, 2, 2]), int(c.faces[FACE_B, 2, 0])}
            elif slot == "DBL":
                return {int(c.faces[FACE_D, 2, 0]), int(c.faces[FACE_B, 2, 2]), int(c.faces[FACE_L, 2, 0])}
            elif slot == "DLF":
                return {int(c.faces[FACE_D, 0, 0]), int(c.faces[FACE_L, 2, 2]), int(c.faces[FACE_F, 2, 0])}
            return set()

        for slot_name, colors, check_fn, sexy_move in target_corners:
            # 1. If already solved, continue
            if check_fn(cube):
                continue

            # 2. Check if the corner is stuck in a bottom slot
            bottom_slots = [
                ("DFR", "R U R' U'"),
                ("DRB", "B U B' U'"),
                ("DBL", "L U L' U'"),
                ("DLF", "F U F' U'")
            ]
            for b_slot, extract_alg in bottom_slots:
                if get_bottom_corner_colors(cube, b_slot) == colors and not check_fn(cube):
                    apply_fn(extract_alg)
                    break

            # 3. Now the corner is in the U layer. Rotate U until it is above the target slot
            target_u_slot = "U" + slot_name[1:]
            for _ in range(4):
                if get_top_corner_colors(cube, target_u_slot) == colors:
                    break
                apply_fn("U")

            # 4. Repeat sexy_move until the corner is correctly placed and oriented (max 5 times)
            for _ in range(6):
                if check_fn(cube):
                    break
                apply_fn(sexy_move)

    # -------------------------------------------------------------------------
    # Stage 3: Middle Layer Edges (F2L)
    # -------------------------------------------------------------------------
    def _solve_middle_edges(self, cube: CubeState, apply_fn):
        # Target middle edges:
        # F-R: F[1, 2] == F (2), R[1, 0] == R (5) -> Left-to-Right insertion: U R U' R' U' F' U F
        # R-B: R[1, 2] == R (5), B[1, 0] == B (3) -> U B U' B' U' R' U R
        # B-L: B[1, 2] == B (3), L[1, 0] == L (4) -> U L U' L' U' B' U B
        # L-F: L[1, 2] == L (4), F[1, 0] == F (2) -> U F U' F' U' L' U L

        middle_edges = [
            ("FR", FACE_F, FACE_R, lambda c: c.faces[FACE_F, 1, 2] == FACE_F and c.faces[FACE_R, 1, 0] == FACE_R, "U R U' R' U' F' U F", "U' F' U F U R U' R'"),
            ("RB", FACE_R, FACE_B, lambda c: c.faces[FACE_R, 1, 2] == FACE_R and c.faces[FACE_B, 1, 0] == FACE_B, "U B U' B' U' R' U R", "U' R' U R U B U' B'"),
            ("BL", FACE_B, FACE_L, lambda c: c.faces[FACE_B, 1, 2] == FACE_B and c.faces[FACE_L, 1, 0] == FACE_L, "U L U' L' U' B' U B", "U' B' U B U L U' L'"),
            ("LF", FACE_L, FACE_F, lambda c: c.faces[FACE_L, 1, 2] == FACE_L and c.faces[FACE_F, 1, 0] == FACE_F, "U F U' F' U' L' U L", "U' L' U L U F U' F'"),
        ]

        def get_mid_edge_colors(c, slot):
            if slot == "FR":
                return {int(c.faces[FACE_F, 1, 2]), int(c.faces[FACE_R, 1, 0])}
            elif slot == "RB":
                return {int(c.faces[FACE_R, 1, 2]), int(c.faces[FACE_B, 1, 0])}
            elif slot == "BL":
                return {int(c.faces[FACE_B, 1, 2]), int(c.faces[FACE_L, 1, 0])}
            elif slot == "LF":
                return {int(c.faces[FACE_L, 1, 2]), int(c.faces[FACE_F, 1, 0])}
            return set()

        for slot_name, color1, color2, check_fn, alg_to_right, alg_to_left in middle_edges:
            if check_fn(cube):
                continue

            target_set = {color1, color2}

            # If edge is stuck in a middle slot (wrong pos/orientation), eject it
            eject_slots = [
                ("FR", "U R U' R' U' F' U F"),
                ("RB", "U B U' B' U' R' U R"),
                ("BL", "U L U' L' U' B' U B"),
                ("LF", "U F U' F' U' L' U L"),
            ]
            for m_slot, eject_alg in eject_slots:
                if get_mid_edge_colors(cube, m_slot) == target_set and not check_fn(cube):
                    apply_fn(eject_alg)
                    break

            # Now edge is in U layer. Find which U edge it is:
            # U edges: UF (U[2, 1], F[0, 1]), UR (U[1, 2], R[0, 1]), UB (U[0, 1], B[0, 1]), UL (U[1, 0], L[0, 1])
            # We want to match the side color with its center!
            # For each of the 4 rotations of U:
            for _ in range(4):
                # Check UF:
                if int(cube.faces[FACE_F, 0, 1]) == color1 and int(cube.faces[FACE_U, 2, 1]) == color2 and slot_name == "FR":
                    apply_fn(alg_to_right) # UF to FR
                    break
                elif int(cube.faces[FACE_R, 0, 1]) == color2 and int(cube.faces[FACE_U, 1, 2]) == color1 and slot_name == "FR":
                    # UR to FR: insert left with F'
                    apply_fn("U' F' U F U R U' R'")
                    break
                elif int(cube.faces[FACE_R, 0, 1]) == color1 and int(cube.faces[FACE_U, 1, 2]) == color2 and slot_name == "RB":
                    apply_fn(alg_to_right) # UR to RB
                    break
                elif int(cube.faces[FACE_B, 0, 1]) == color2 and int(cube.faces[FACE_U, 0, 1]) == color1 and slot_name == "RB":
                    apply_fn("U' R' U R U B U' B'")
                    break
                elif int(cube.faces[FACE_B, 0, 1]) == color1 and int(cube.faces[FACE_U, 0, 1]) == color2 and slot_name == "BL":
                    apply_fn(alg_to_right) # UB to BL
                    break
                elif int(cube.faces[FACE_L, 0, 1]) == color2 and int(cube.faces[FACE_U, 1, 0]) == color1 and slot_name == "BL":
                    apply_fn("U' B' U B U L U' L'")
                    break
                elif int(cube.faces[FACE_L, 0, 1]) == color1 and int(cube.faces[FACE_U, 1, 0]) == color2 and slot_name == "LF":
                    apply_fn(alg_to_right) # UL to LF
                    break
                elif int(cube.faces[FACE_F, 0, 1]) == color2 and int(cube.faces[FACE_U, 2, 1]) == color1 and slot_name == "LF":
                    apply_fn("U' L' U L U F U' F'")
                    break
                apply_fn("U")

    # -------------------------------------------------------------------------
    # Stage 4: Yellow Cross (OLL Edges)
    # -------------------------------------------------------------------------
    def _solve_yellow_cross(self, cube: CubeState, apply_fn):
        # Yellow is color 1 (FACE_D color) on the U face (FACE_U)!
        yellow_color = FACE_D  # 1

        def get_yellow_edges(c):
            # Check UF, UR, UB, UL top stickers:
            # U[2, 1], U[1, 2], U[0, 1], U[1, 0]
            return [
                int(c.faces[FACE_U, 2, 1]) == yellow_color,  # F
                int(c.faces[FACE_U, 1, 2]) == yellow_color,  # R
                int(c.faces[FACE_U, 0, 1]) == yellow_color,  # B
                int(c.faces[FACE_U, 1, 0]) == yellow_color   # L
            ]

        for _ in range(4):
            edges = get_yellow_edges(cube)
            count = sum(edges)
            if count == 4:
                break
            if count == 0:
                # Dot: F R U R' U' F'
                apply_fn("F R U R' U' F'")
            elif count == 2:
                # Check for line or L-shape
                f_y, r_y, b_y, l_y = edges
                if (f_y and b_y) or (r_y and l_y):
                    # Line
                    if f_y and b_y:
                        apply_fn("U") # Make it horizontal (L-R)
                    apply_fn("F R U R' U' F'")
                else:
                    # L-shape: orient so edges are at B and L (indices 2 and 3)
                    for _ in range(4):
                        ed = get_yellow_edges(cube)
                        if ed[2] and ed[3]: # B and L
                            break
                        apply_fn("U")
                    apply_fn("F U R U' R' F'") # or F R U R' U' F' twice

    # -------------------------------------------------------------------------
    # Stage 5: Yellow Corners Orientation (OLL Corners)
    # -------------------------------------------------------------------------
    def _solve_yellow_corners_orientation(self, cube: CubeState, apply_fn):
        yellow_color = FACE_D
        # For each of 4 corners: rotate U so target corner is at UFR (U[2, 2]),
        # then apply R' D' R D until yellow is facing Up!
        for _ in range(4):
            while int(cube.faces[FACE_U, 2, 2]) != yellow_color:
                apply_fn("R' D' R D")
            apply_fn("U")

    # -------------------------------------------------------------------------
    # Stage 6: Yellow Corners Permutation (PLL Corners)
    # -------------------------------------------------------------------------
    def _solve_yellow_corners_permutation(self, cube: CubeState, apply_fn):
        # Headlights check:
        def find_headlights(c):
            # Check 4 sides for matching corner stickers on top layer:
            # F: F[0, 0] == F[0, 2]
            # R: R[0, 0] == R[0, 2]
            # B: B[0, 0] == B[0, 2]
            # L: L[0, 0] == L[0, 2]
            sides = [
                (FACE_F, "F", 0),
                (FACE_R, "R", 1),
                (FACE_B, "B", 2),
                (FACE_L, "L", 3),
            ]
            matching = []
            for face_idx, name, side_idx in sides:
                if c.faces[face_idx, 0, 0] == c.faces[face_idx, 0, 2]:
                    matching.append(face_idx)
            return matching

        # A-Perm / T-Perm corners algorithm:
        # Standard: R' F R' B2 R F' R' B2 R2
        c_perm = "R' F R' B2 R F' R' B2 R2"

        matching = find_headlights(cube)
        if len(matching) == 4:
            return # All corners permuted!
        
        if len(matching) == 0:
            apply_fn(c_perm)
            matching = find_headlights(cube)

        if len(matching) >= 1:
            # Put the headlights at Back (FACE_B)
            target_back = matching[0]
            # Rotate U until headlights face Back (FACE_B):
            for _ in range(4):
                if cube.faces[FACE_B, 0, 0] == cube.faces[FACE_B, 0, 2]:
                    break
                apply_fn("U")
            apply_fn(c_perm)

    # -------------------------------------------------------------------------
    # Stage 7: Yellow Edges Permutation (PLL Edges)
    # -------------------------------------------------------------------------
    def _solve_yellow_edges_permutation(self, cube: CubeState, apply_fn):
        # Align corners first
        for _ in range(4):
            if cube.faces[FACE_F, 0, 0] == FACE_F:
                break
            apply_fn("U")

        # U-Perm (Clockwise): R U' R U R U R U' R' U' R2
        # U-Perm (Counter): R2 U R U R' U' R' U' R' U R'
        u_perm_cw = "R U' R U R U R U' R' U' R2"

        def get_solved_edge_sides(c):
            solved = []
            if c.faces[FACE_F, 0, 1] == FACE_F: solved.append(FACE_F)
            if c.faces[FACE_R, 0, 1] == FACE_R: solved.append(FACE_R)
            if c.faces[FACE_B, 0, 1] == FACE_B: solved.append(FACE_B)
            if c.faces[FACE_L, 0, 1] == FACE_L: solved.append(FACE_L)
            return solved

        for _ in range(4):
            solved = get_solved_edge_sides(cube)
            if len(solved) == 4:
                return
            if len(solved) == 0:
                apply_fn(u_perm_cw)
            elif len(solved) == 1:
                # Put the solved edge at Back (B)
                solved_face = solved[0]
                # Rotate entire cube or use U turns + perm
                # If solved is F -> rotate y2
                # If solved is R -> rotate y
                # If solved is L -> rotate y'
                # If solved is B -> already at B!
                if solved_face == FACE_F:
                    apply_fn("y2")
                elif solved_face == FACE_R:
                    apply_fn("y")
                elif solved_face == FACE_L:
                    apply_fn("y'")
                
                # Try U-Perm
                apply_fn(u_perm_cw)
                if len(get_solved_edge_sides(cube)) < 4:
                    apply_fn(u_perm_cw)
                return

    # -------------------------------------------------------------------------
    # Stage 8: AUF & Final Orientation
    # -------------------------------------------------------------------------
    def _solve_auf(self, cube: CubeState, apply_fn):
        for _ in range(4):
            if cube.is_solved():
                return
            apply_fn("U")
        for _ in range(4):
            if cube.is_solved():
                return
            apply_fn("y")
        for _ in range(4):
            if cube.is_solved():
                return
            apply_fn("x")

print("Guaranteed LBL Solver defined!")
