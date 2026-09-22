"""
Astra-DeepCube: Scramble Generator
Generates WCA-compliant scramble sequences and depth-controlled training scrambles.
"""

import random
from typing import List


STANDARD_MOVES = ["U", "D", "L", "R", "F", "B"]
MODIFIERS = ["", "'", "2"]

WIDE_MOVES_4X4 = ["U", "D", "L", "R", "F", "B", "Uw", "Dw", "Lw", "Rw", "Fw", "Bw"]
WIDE_MOVES_5X5 = ["U", "D", "L", "R", "F", "B", "Uw", "Dw", "Lw", "Rw", "Fw", "Bw", "2U", "2D", "2L", "2R", "2F", "2B"]

OPPOSITE_FACES = {
    "U": "D", "D": "U",
    "L": "R", "R": "L",
    "F": "B", "B": "F"
}


def generate_scramble(length: int = 20, size: int = 3) -> str:
    """
    Generates a valid scramble sequence where consecutive moves do not cancel out
    (e.g., avoids U followed by U or U').
    """
    if size == 3:
        move_pool = STANDARD_MOVES
    elif size == 4:
        move_pool = WIDE_MOVES_4X4
    else:
        move_pool = WIDE_MOVES_5X5

    moves: List[str] = []
    last_face = ""
    second_last_face = ""

    for _ in range(length):
        valid_candidates = []
        for base in move_pool:
            base_char = base[0]
            # Avoid same face consecutively
            if base_char == last_face:
                continue
            # Avoid redundant opposing face sequences (e.g., U D U)
            if base_char == second_last_face and OPPOSITE_FACES.get(last_face) == base_char:
                continue
            valid_candidates.append(base)

        chosen_base = random.choice(valid_candidates)
        modifier = random.choice(MODIFIERS)
        
        move_str = chosen_base + modifier
        moves.append(move_str)

        second_last_face = last_face
        last_face = chosen_base[0]

    return " ".join(moves)


def invert_move(move: str) -> str:
    """Computes the inverse of a given move (e.g., R -> R', R' -> R, R2 -> R2)."""
    move = move.strip()
    if not move:
        return ""
    if move.endswith("2"):
        return move
    if move.endswith("'"):
        return move[:-1]
    return move + "'"


def invert_moves(moves: str) -> str:
    """Computes the inverse sequence of moves in reverse order."""
    move_list = moves.strip().split()
    inverted = [invert_move(m) for m in reversed(move_list)]
    return " ".join(inverted)
