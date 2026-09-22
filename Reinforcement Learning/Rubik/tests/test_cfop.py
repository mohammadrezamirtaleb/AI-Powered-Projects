import sys
sys.path.insert(0, ".")
from core.cube_state import CubeState, FACE_U, FACE_D, FACE_F, FACE_B, FACE_L, FACE_R
from core.scrambler import generate_scramble, invert_move
from collections import deque


def get_cross_solved_count(cube: CubeState) -> int:
    """Checks how many of the 4 D-edges are correctly placed and oriented."""
    cnt = 0
    # D-F edge: D[0, 1] == D and F[2, 1] == F
    if cube.faces[FACE_D, 0, 1] == FACE_D and cube.faces[FACE_F, 2, 1] == FACE_F:
        cnt += 1
    # D-R edge: D[1, 2] == D and R[2, 1] == R
    if cube.faces[FACE_D, 1, 2] == FACE_D and cube.faces[FACE_R, 2, 1] == FACE_R:
        cnt += 1
    # D-B edge: D[2, 1] == D and B[2, 1] == B
    if cube.faces[FACE_D, 2, 1] == FACE_D and cube.faces[FACE_B, 2, 1] == FACE_B:
        cnt += 1
    # D-L edge: D[1, 0] == D and L[2, 1] == L
    if cube.faces[FACE_D, 1, 0] == FACE_D and cube.faces[FACE_L, 2, 1] == FACE_L:
        cnt += 1
    return cnt


print("Testing cross helper...")
c = CubeState(3)
print(f"Solved cube cross: {get_cross_solved_count(c)} / 4")
