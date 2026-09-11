# City Simulation Architecture

## Purpose
The city layer is a dual-node road network, not a single 4-way box. It models two east-west arterials connected by a north-south corridor, with a roundabout as the southern hub.

## Spatial graph

```
        [North spawn]
              |
     3+3 lane NS arterial
              |
     +--------+--------+   Main Avenue (EW arterial, signalized)
     |   4-way node    |
     +--------+--------+
              |
        connector NS
              |
     +----( roundabout )----+   Boulevard (second EW street)
     |         O            |
     +----------------------+
              |
        [South spawn]
```

## Nodes
| Node | Control | Role |
|------|---------|------|
| Main 4-way | Pre-timed / adaptive signals | Primary conflict box, pedestrian zebras, stop lines |
| Roundabout | Yield-on-entry, circulating CCW | Connects boulevard to NS corridor |
| Connector | Unsignalized through lanes | Links the two EW streets |

## Lane model
- `NUM_LANES_PER_DIR = 3` (right / through / left)
- `LANE_WIDTH = 26 px`, two-way `ROAD_WIDTH = 156 px`
- Right-hand traffic: incoming N uses `x < cx`, incoming S uses `x > cx`

## Route classes
- `STRAIGHT` / `LEFT` / `RIGHT`: cubic approach at the signalized node
- `ROUNDABOUT`: polyline + circulating arc (`alpha` decreases: N → W → S → E)
- `THROUGH`: multi-node path (main stop line, then yield into the circle, then exit)

## Geometry owners
- `src/simulation/intersection.py` — graph, lanes, arc-length paths
- `src/render/renderer.py` — asphalt, island, boulevard, 3-lane dashes
- `src/config.py` — world constants

## Invariants
1. Roundabout island radius < circulating radius < outer curb.
2. Main stop lines sit 36 px upstream of the junction box.
3. Boulevard centerline is `rby`; main avenue centerline is `cy`.
4. Resize recenters `cx` and places `cy` at 34% and `rby` at 74% of canvas height.

## Verification
- Route count > 12 (legacy single intersection).
- Every route `total_length > 0` and `get_pose_at_distance(0)` is finite.
- Vehicles spawned on boulevard routes pass within `r_outer + 20` of the island.
