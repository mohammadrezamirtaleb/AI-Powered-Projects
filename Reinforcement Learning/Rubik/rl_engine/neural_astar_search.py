"""
Astra-DeepCube: Neural Weighted A* Search Engine & Live Telemetry
Drives DeepCubeA heuristic search with batched neighbor inference and deterministic tie-breaking.
"""

import heapq
import time
from typing import List, Dict, Tuple, Optional, Set, Callable
import numpy as np

from core.cube_state import CubeState
from rl_engine.deepcube_model import DeepCubeNetwork
from core.scrambler import invert_move


ACTION_LIST = ["U", "U'", "D", "D'", "L", "L'", "R", "R'", "F", "F'", "B", "B'"]


class SearchNode:
    """Represents a node in the Neural A* Priority Queue."""
    __slots__ = ('f', 'g', 'h', 'cube', 'path', 'q_values', 'node_id')

    def __init__(self, f: float, g: int, h: float, cube: CubeState, path: List[str], q_values: np.ndarray, node_id: int):
        self.f = f
        self.g = g
        self.h = h
        self.cube = cube
        self.path = path
        self.q_values = q_values
        self.node_id = node_id

    def __lt__(self, other: "SearchNode") -> bool:
        if abs(self.f - other.f) > 1e-5:
            return self.f < other.f
        if abs(self.h - other.h) > 1e-5:
            return self.h < other.h
        return self.node_id < other.node_id


class NeuralAStarSolver:
    """
    Weighted Neural A* Search Engine with batched inference and live telemetry.
    """

    def __init__(self, model: DeepCubeNetwork, weight_lambda: float = 1.2):
        self.model = model
        self.weight_lambda = weight_lambda

    def state_to_key(self, cube: CubeState) -> bytes:
        """Returns compact byte hash of cube state."""
        return cube.faces.tobytes()

    def solve(
        self,
        initial_cube: CubeState,
        max_nodes: int = 5000,
        timeout_sec: float = 10.0,
        telemetry_callback: Optional[Callable[[Dict], None]] = None
    ) -> Tuple[Optional[List[str]], Dict]:
        """
        Executes Weighted Neural A* Search with batched inference.
        """
        if initial_cube.is_solved():
            return [], {"nodes_expanded": 0, "time_sec": 0.0, "status": "SOLVED_INSTANT"}

        start_time = time.time()
        initial_h, initial_q = self.model.predict_single(initial_cube.to_one_hot_tensor())
        
        root = SearchNode(
            f=self.weight_lambda * initial_h,
            g=0,
            h=initial_h,
            cube=initial_cube.clone(),
            path=[],
            q_values=initial_q,
            node_id=0
        )

        open_heap: List[SearchNode] = [root]
        closed_set: Set[bytes] = set()
        
        nodes_expanded = 0
        min_h_seen = initial_h
        node_counter = 1

        last_telemetry_time = time.time()

        BATCH_SIZE = 128

        while open_heap and nodes_expanded < max_nodes:
            elapsed = time.time() - start_time
            if elapsed > timeout_sec:
                break

            current_nodes = []
            while open_heap and len(current_nodes) < BATCH_SIZE:
                current = heapq.heappop(open_heap)
                state_key = self.state_to_key(current.cube)

                if state_key in closed_set:
                    continue
                closed_set.add(state_key)
                current_nodes.append(current)

            if not current_nodes:
                continue

            nodes_expanded += len(current_nodes)

            # Use best node in batch for telemetry tracking
            best_current = current_nodes[0]
            if best_current.h < min_h_seen:
                min_h_seen = best_current.h

            if best_current.cube.is_solved():
                stats = {
                    "nodes_expanded": nodes_expanded,
                    "queue_size": len(open_heap),
                    "time_sec": round(time.time() - start_time, 4),
                    "path_length": len(best_current.path),
                    "status": "SUCCESS",
                    "min_h": min_h_seen
                }
                return best_current.path, stats

            # Exception-safe telemetry update (~30 FPS)
            if telemetry_callback and (time.time() - last_telemetry_time > 0.03):
                try:
                    telemetry_callback({
                        "nodes_expanded": nodes_expanded,
                        "queue_size": len(open_heap),
                        "current_g": best_current.g,
                        "current_h": round(best_current.h, 2),
                        "current_f": round(best_current.f, 2),
                        "min_h": round(min_h_seen, 2),
                        "active_path": " ".join(best_current.path[-6:]),
                        "q_values": best_current.q_values.tolist() if hasattr(best_current.q_values, "tolist") else list(best_current.q_values),
                        "elapsed_sec": round(time.time() - start_time, 2),
                    })
                except Exception:
                    pass
                last_telemetry_time = time.time()

            candidate_neighbors = []
            candidate_tensors = []

            for current in current_nodes:
                last_move = current.path[-1] if current.path else ""
                last_face = last_move[0] if last_move else ""

                for action_idx, action in enumerate(ACTION_LIST):
                    base_face = action[0]
                    if base_face == last_face and (action == invert_move(last_move)):
                        continue

                    neighbor_cube = current.cube.clone()
                    neighbor_cube.apply_move(action)
                    neighbor_key = self.state_to_key(neighbor_cube)

                    if neighbor_key in closed_set:
                        continue

                    if neighbor_cube.is_solved():
                        sol_path = current.path + [action]
                        stats = {
                            "nodes_expanded": nodes_expanded + 1,
                            "queue_size": len(open_heap),
                            "time_sec": round(time.time() - start_time, 4),
                            "path_length": len(sol_path),
                            "status": "SUCCESS",
                            "min_h": 0.0
                        }
                        return sol_path, stats

                    candidate_neighbors.append((current, neighbor_cube, action))
                    candidate_tensors.append(neighbor_cube.to_one_hot_tensor())

            # Batched forward pass over ALL valid neighbors of ALL nodes in batch
            if candidate_tensors:
                h_vals, q_probs_list = self.model.predict_batch(candidate_tensors)
                
                for idx, (parent_node, nbr_cube, act) in enumerate(candidate_neighbors):
                    h_val = h_vals[idx]
                    q_probs = q_probs_list[idx]
                    g_val = parent_node.g + 1
                    f_val = g_val + self.weight_lambda * h_val

                    neighbor_node = SearchNode(
                        f=f_val,
                        g=g_val,
                        h=h_val,
                        cube=nbr_cube,
                        path=parent_node.path + [act],
                        q_values=q_probs,
                        node_id=node_counter
                    )
                    node_counter += 1
                    heapq.heappush(open_heap, neighbor_node)

        stats = {
            "nodes_expanded": nodes_expanded,
            "queue_size": len(open_heap),
            "time_sec": round(time.time() - start_time, 4),
            "status": "TIMEOUT_OR_EXHAUSTED",
            "min_h": min_h_seen
        }
        return None, stats
