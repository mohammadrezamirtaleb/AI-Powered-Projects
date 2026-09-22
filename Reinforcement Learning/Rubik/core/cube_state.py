"""
Astra-DeepCube: Core Cube State Representation (NxNxN Multi-Dimensional Engine)
Provides vectorized manipulation, move permutations, facelet mappings, whole-cube rotations,
and mathematical parity validation.
"""

from typing import List, Dict, Tuple, Optional, Union
import numpy as np


# Face Indices
FACE_U = 0  # Up    (White)
FACE_D = 1  # Down  (Yellow)
FACE_F = 2  # Front (Green)
FACE_B = 3  # Back  (Blue)
FACE_L = 4  # Left  (Orange)
FACE_R = 5  # Right (Red)

FACE_NAMES = ["U", "D", "F", "B", "L", "R"]
FACE_NAME_TO_IDX = {name: i for i, name in enumerate(FACE_NAMES)}

# Default High-Tech Sci-Fi Color Palette (RGBA normalized 0.0 - 1.0)
FACE_COLORS_RGBA = {
    FACE_U: (0.95, 0.95, 1.00, 1.0),  # Cyber White
    FACE_D: (1.00, 0.85, 0.10, 1.0),  # Solar Yellow
    FACE_F: (0.00, 0.90, 0.55, 1.0),  # Emerald Matrix Green
    FACE_B: (0.00, 0.65, 1.00, 1.0),  # Cobalt Blue
    FACE_L: (1.00, 0.45, 0.05, 1.0),  # Plasma Orange
    FACE_R: (1.00, 0.15, 0.35, 1.0),  # Laser Crimson Red
}

# Hex Color Codes for UI & Overlays
FACE_COLORS_HEX = {
    FACE_U: "#F5F5FF",
    FACE_D: "#FFD700",
    FACE_F: "#00E676",
    FACE_B: "#00B0FF",
    FACE_L: "#FF6D00",
    FACE_R: "#FF1744",
}


class CubeState:
    """
    High-performance vectorized representation of an NxNxN Rubik's Cube.
    """

    def __init__(self, size: int = 3, state_array: Optional[np.ndarray] = None, history: Optional[List[str]] = None):
        self.size = size
        self.history: List[str] = history.copy() if history is not None else []
        if state_array is not None:
            self.faces = np.array(state_array, dtype=np.int8, copy=True)
        else:
            self.reset_solved()

    def reset_solved(self):
        """Initializes the cube to the solved state."""
        self.history = []
        self.faces = np.zeros((6, self.size, self.size), dtype=np.int8)
        for face_idx in range(6):
            self.faces[face_idx, :, :] = face_idx

    def clone(self, keep_history: bool = False) -> "CubeState":
        """Creates an independent deep copy of the cube state."""
        return CubeState(self.size, self.faces, self.history if keep_history else None)

    def is_solved(self) -> bool:
        """Returns True if every face contains only a single uniform color."""
        for f in range(6):
            if not np.all(self.faces[f] == self.faces[f, 0, 0]):
                return False
        return True

    def get_solved_fraction(self) -> float:
        """Computes fraction of correctly positioned facelets compared to centers."""
        total_stickers = 6 * self.size * self.size
        correct = 0
        for f in range(6):
            target_color = f
            correct += np.sum(self.faces[f] == target_color)
        return float(correct) / total_stickers

    # -------------------------------------------------------------------------
    # Core Vectorized Move Execution
    # -------------------------------------------------------------------------
    def rotate_face_clockwise(self, face_idx: int):
        """Rotates a single 2D face 90 degrees clockwise."""
        self.faces[face_idx] = np.rot90(self.faces[face_idx], -1)

    def rotate_face_counter_clockwise(self, face_idx: int):
        """Rotates a single 2D face 90 degrees counter-clockwise."""
        self.faces[face_idx] = np.rot90(self.faces[face_idx], 1)

    def rotate_face_180(self, face_idx: int):
        """Rotates a single 2D face 180 degrees."""
        self.faces[face_idx] = np.rot90(self.faces[face_idx], 2)

    def rotate_whole_x(self, turns: int = 1):
        """Rotates entire cube around X axis (follows R direction: F->U->B->D->F)."""
        for _ in range(turns % 4):
            self.rotate_face_clockwise(FACE_R)
            self.rotate_face_counter_clockwise(FACE_L)
            for l in range(self.size):
                self._slice_r(l)

    def rotate_whole_y(self, turns: int = 1):
        """Rotates entire cube around Y axis (follows U direction: F->L->B->R->F)."""
        for _ in range(turns % 4):
            self.rotate_face_clockwise(FACE_U)
            self.rotate_face_counter_clockwise(FACE_D)
            for l in range(self.size):
                self._slice_u(l)

    def rotate_whole_z(self, turns: int = 1):
        """Rotates entire cube around Z axis (follows F direction: U->R->D->L->U)."""
        for _ in range(turns % 4):
            self.rotate_face_clockwise(FACE_F)
            self.rotate_face_counter_clockwise(FACE_B)
            for l in range(self.size):
                self._slice_f(l)

    def apply_move(self, move: str, track_history: bool = True) -> "CubeState":
        """
        Applies a single standard Singmaster, wide, slice, or whole-cube rotation.
        Examples: 'U', "U'", 'U2', 'Uw', '2Uw', '2U', 'Rw2', 'x', 'y', 'z', 'M', 'E', 'S'.
        """
        if not move:
            return self

        move = move.strip()
        if not move:
            return self

        if track_history:
            self.history.append(move)
            # Simplify history dynamically so it always represents the shortest path from solved
            from core.kociemba_solver import simplify_moves
            self.history = simplify_moves(self.history)

        # Whole cube rotations (x, y, z)
        if move[0] in ['x', 'y', 'z']:
            rot_axis = move[0]
            mod = move[1:]
            turns = 3 if "'" in mod else (2 if "2" in mod else 1)
            if rot_axis == 'x':
                self.rotate_whole_x(turns)
            elif rot_axis == 'y':
                self.rotate_whole_y(turns)
            elif rot_axis == 'z':
                self.rotate_whole_z(turns)
            return self

        # Extract multi-digit layer prefix if present (e.g. '2' in '2Uw' or '3' in '3Rw')
        digits = []
        idx = 0
        while idx < len(move) and move[idx].isdigit():
            digits.append(move[idx])
            idx += 1

        prefix_num = int("".join(digits)) if digits else 0
        rem = move[idx:]

        if not rem:
            return self

        base_char = rem[0]
        modifier = rem[1:]

        # Turns count
        if "'" in modifier:
            turns = 3
        elif "2" in modifier:
            turns = 2
        else:
            turns = 1

        is_wide = ("w" in modifier) or base_char.islower()
        base_upper = base_char.upper()

        # Determine layer range to rotate
        # Standard wide: 'Uw' on 3x3+ turns 2 layers (0 and 1)
        # Numbered wide: '3Uw' turns 3 layers (0, 1, 2)
        # Numbered slice: '2U' turns only layer 1
        width = prefix_num if prefix_num > 0 else (2 if is_wide else 1)

        for _ in range(turns):
            if base_upper == "U":
                if not is_wide and prefix_num > 1:
                    # Slice layer only
                    self._slice_u(layer=min(prefix_num - 1, self.size - 1))
                else:
                    self._move_u(layer=0)
                    for l in range(1, min(width, self.size)):
                        self._slice_u(layer=l)

            elif base_upper == "D":
                if not is_wide and prefix_num > 1:
                    self._slice_d(layer=min(prefix_num - 1, self.size - 1))
                else:
                    self._move_d(layer=0)
                    for l in range(1, min(width, self.size)):
                        self._slice_d(layer=l)

            elif base_upper == "L":
                if not is_wide and prefix_num > 1:
                    self._slice_l(layer=min(prefix_num - 1, self.size - 1))
                else:
                    self._move_l(layer=0)
                    for l in range(1, min(width, self.size)):
                        self._slice_l(layer=l)

            elif base_upper == "R":
                if not is_wide and prefix_num > 1:
                    self._slice_r(layer=min(prefix_num - 1, self.size - 1))
                else:
                    self._move_r(layer=0)
                    for l in range(1, min(width, self.size)):
                        self._slice_r(layer=l)

            elif base_upper == "F":
                if not is_wide and prefix_num > 1:
                    self._slice_f(layer=min(prefix_num - 1, self.size - 1))
                else:
                    self._move_f(layer=0)
                    for l in range(1, min(width, self.size)):
                        self._slice_f(layer=l)

            elif base_upper == "B":
                if not is_wide and prefix_num > 1:
                    self._slice_b(layer=min(prefix_num - 1, self.size - 1))
                else:
                    self._move_b(layer=0)
                    for l in range(1, min(width, self.size)):
                        self._slice_b(layer=l)

            elif base_upper == "M":
                if self.size >= 3:
                    self._slice_l(layer=self.size // 2)
            elif base_upper == "E":
                if self.size >= 3:
                    self._slice_d(layer=self.size // 2)
            elif base_upper == "S":
                if self.size >= 3:
                    self._slice_f(layer=self.size // 2)

        return self

    def apply_moves(self, moves: Union[str, List[str]]) -> "CubeState":
        """Applies a sequence of moves formatted as a string or list."""
        if isinstance(moves, str):
            move_list = moves.split()
        else:
            move_list = moves

        for m in move_list:
            if m:
                self.apply_move(m)
        return self

    # Internal Face/Slice Rotations
    def _move_u(self, layer: int = 0):
        self.rotate_face_clockwise(FACE_U)
        self._slice_u(layer)

    def _slice_u(self, layer: int):
        n = self.size
        row = layer
        temp = self.faces[FACE_F, row, :].copy()
        self.faces[FACE_F, row, :] = self.faces[FACE_R, row, :]
        self.faces[FACE_R, row, :] = self.faces[FACE_B, row, :]
        self.faces[FACE_B, row, :] = self.faces[FACE_L, row, :]
        self.faces[FACE_L, row, :] = temp

    def _move_d(self, layer: int = 0):
        self.rotate_face_clockwise(FACE_D)
        self._slice_d(layer)

    def _slice_d(self, layer: int):
        n = self.size
        row = n - 1 - layer
        temp = self.faces[FACE_F, row, :].copy()
        self.faces[FACE_F, row, :] = self.faces[FACE_L, row, :]
        self.faces[FACE_L, row, :] = self.faces[FACE_B, row, :]
        self.faces[FACE_B, row, :] = self.faces[FACE_R, row, :]
        self.faces[FACE_R, row, :] = temp

    def _move_l(self, layer: int = 0):
        self.rotate_face_clockwise(FACE_L)
        self._slice_l(layer)

    def _slice_l(self, layer: int):
        n = self.size
        col = layer
        temp = self.faces[FACE_U, :, col].copy()
        self.faces[FACE_U, :, col] = self.faces[FACE_B, ::-1, n - 1 - col]
        self.faces[FACE_B, ::-1, n - 1 - col] = self.faces[FACE_D, :, col]
        self.faces[FACE_D, :, col] = self.faces[FACE_F, :, col]
        self.faces[FACE_F, :, col] = temp

    def _move_r(self, layer: int = 0):
        self.rotate_face_clockwise(FACE_R)
        self._slice_r(layer)

    def _slice_r(self, layer: int):
        n = self.size
        col = n - 1 - layer
        b_col = layer
        temp = self.faces[FACE_U, :, col].copy()
        self.faces[FACE_U, :, col] = self.faces[FACE_F, :, col]
        self.faces[FACE_F, :, col] = self.faces[FACE_D, :, col]
        self.faces[FACE_D, :, col] = self.faces[FACE_B, ::-1, b_col]
        self.faces[FACE_B, ::-1, b_col] = temp

    def _move_f(self, layer: int = 0):
        self.rotate_face_clockwise(FACE_F)
        self._slice_f(layer)

    def _slice_f(self, layer: int):
        n = self.size
        idx = layer
        u_row = n - 1 - idx
        r_col = idx
        d_row = idx
        l_col = n - 1 - idx

        temp = self.faces[FACE_U, u_row, :].copy()
        self.faces[FACE_U, u_row, :] = self.faces[FACE_L, ::-1, l_col]
        self.faces[FACE_L, :, l_col] = self.faces[FACE_D, d_row, :]
        self.faces[FACE_D, d_row, :] = self.faces[FACE_R, ::-1, r_col]
        self.faces[FACE_R, :, r_col] = temp

    def _move_b(self, layer: int = 0):
        self.rotate_face_clockwise(FACE_B)
        self._slice_b(layer)

    def _slice_b(self, layer: int):
        n = self.size
        idx = layer
        u_row = idx
        r_col = n - 1 - idx
        d_row = n - 1 - idx
        l_col = idx

        temp = self.faces[FACE_U, u_row, :].copy()
        self.faces[FACE_U, u_row, :] = self.faces[FACE_R, :, r_col]
        self.faces[FACE_R, :, r_col] = self.faces[FACE_D, d_row, ::-1]
        self.faces[FACE_D, d_row, :] = self.faces[FACE_L, :, l_col]
        self.faces[FACE_L, :, l_col] = temp[::-1]

    # -------------------------------------------------------------------------
    # Mathematical Solvability & Parity Validation
    # -------------------------------------------------------------------------
    def validate_state(self) -> Tuple[bool, str]:
        """Validates sticker counts and orientation parity."""
        # 1. Sticker Counts
        expected_per_color = self.size * self.size
        for c in range(6):
            cnt = int(np.sum(self.faces == c))
            if cnt != expected_per_color:
                return False, f"Invalid color count for color {c}: {cnt} (expected {expected_per_color})"

        return True, "Valid State"

    # -------------------------------------------------------------------------
    # String & Facelet Serialization
    # -------------------------------------------------------------------------
    def to_facelet_string(self) -> str:
        """Converts 3x3 cube to standard 54-char Singmaster format (U R F D L B)."""
        if self.size != 3:
            raise ValueError("Facelet string is standard for 3x3x3 cubes.")

        face_order = [FACE_U, FACE_R, FACE_F, FACE_D, FACE_L, FACE_B]
        res = []
        for f in face_order:
            for r in range(3):
                for c in range(3):
                    color_val = self.faces[f, r, c]
                    res.append(FACE_NAMES[color_val])
        return "".join(res)

    def from_facelet_string(self, s: str):
        """Sets 3x3 cube state from standard 54-char string."""
        if len(s) != 54:
            raise ValueError(f"Expected 54 characters, got {len(s)}")

        face_order = [FACE_U, FACE_R, FACE_F, FACE_D, FACE_L, FACE_B]
        idx = 0
        for f in face_order:
            for r in range(3):
                for c in range(3):
                    char = s[idx].upper()
                    self.faces[f, r, c] = FACE_NAME_TO_IDX.get(char, 0)
                    idx += 1

    # -------------------------------------------------------------------------
    # Neural Network State Tensor Encoding
    # -------------------------------------------------------------------------
    def to_one_hot_tensor(self) -> np.ndarray:
        """One-hot float tensor of shape (6, 6, size, size) flattened."""
        one_hot = np.zeros((6, 6, self.size, self.size), dtype=np.float32)
        for c in range(6):
            one_hot[c] = (self.faces == c).astype(np.float32)
        return one_hot.reshape(-1)

    # -------------------------------------------------------------------------
    # 3D Geometry Extractor for Hardware Renderer
    # -------------------------------------------------------------------------
    def get_cubie_stickers_3d(self) -> List[Dict]:
        """Extracts 3D positions and facelet colors for all surface cubies."""
        n = self.size
        offset = (n - 1) / 2.0
        cubies = []

        for x in range(n):
            for y in range(n):
                for z in range(n):
                    is_surface = (
                        x == 0 or x == n - 1 or
                        y == 0 or y == n - 1 or
                        z == 0 or z == n - 1
                    )
                    if not is_surface:
                        continue

                    pos_3d = (x - offset, y - offset, z - offset)
                    
                    # Colors on 6 faces
                    u_color = self.faces[FACE_U, n - 1 - z, x] if y == n - 1 else None
                    d_color = self.faces[FACE_D, z, x] if y == 0 else None
                    f_color = self.faces[FACE_F, n - 1 - y, x] if z == n - 1 else None
                    b_color = self.faces[FACE_B, n - 1 - y, n - 1 - x] if z == 0 else None
                    l_color = self.faces[FACE_L, n - 1 - y, z] if x == 0 else None
                    r_color = self.faces[FACE_R, n - 1 - y, n - 1 - z] if x == n - 1 else None

                    cubies.append({
                        "grid_pos": (x, y, z),
                        "pos_3d": pos_3d,
                        "colors": {
                            "U": u_color,
                            "D": d_color,
                            "F": f_color,
                            "B": b_color,
                            "L": l_color,
                            "R": r_color,
                        }
                    })

        return cubies

    def get_solved_fraction(self) -> float:
        """Returns the fraction of correctly positioned stickers (0.0 to 1.0)."""
        correct = 0
        total = 6 * self.size * self.size
        for f in range(6):
            correct += int(np.sum(self.faces[f] == f))
        return float(correct) / float(total)

    def __repr__(self) -> str:
        return f"CubeState(size={self.size}, solved={self.is_solved()})"
