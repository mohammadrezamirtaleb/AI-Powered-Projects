# Traffic Simulation Architecture

## Purpose
Produce legally plausible, multi-agent traffic at 60 Hz: signals, queues, pedestrians, weather grip, and a Dueling DQN policy with action repeat.

## Control loops

```
spawners → perception (34D × 3 stack)
        → discrete action (5) × action-repeat k=4
        → kinematics + headway + signals + yield
        → SAT collisions
        → PER / Double DQN (background thread)
```

## Signalized node
Phase ring: `NS_GREEN → NS_YELLOW → ALL_RED → EW_GREEN → EW_YELLOW → ALL_RED`.

- Adaptive mode stretches green when the served queue is heavy and cuts it when empty.
- Emergency preemption only from a stable green, with an 8 s cooldown so yellow does not oscillate.
- Physical stop-line clamp: speed 0 and path lock 4 px before the line on red/yellow (non-emergency).

## Unsignalized roundabout
- Light state for boulevard entries is `YIELD` (not a hard red).
- Yield line is `route.yield_line_dist` on the arc-length path.
- If any live vehicle is in the circulating annulus, entrants brake to a crawl until the gap opens.

## Perception (ML / DL)
Observation `VISION_STATE_SIZE = 34`, stacked `k=3` → 102 inputs:

| Block | Dims |
|-------|------|
| LiDAR range + closing speed | 18 |
| Signal one-hot | 4 |
| Stop, speed, target, leader | 4 |
| Junction density + crossing hazard | 2 |
| Friction | 1 |
| In-circle, circulating density, siren proximity, lane, yield flag | 5 |

Policy: residual Dueling DQN (`320 → 320+skip → 160` then V/A streams), Double DQN + PER, AdamW + ReduceLROnPlateau.

Rewards add courtesy terms: emergency yield, blocking an EMS unit, roundabout gap acceptance.

## Capacity
`MAX_ACTIVE_CARS = 42` with Euclidean spawn clearance (115 px) to protect queue tails.

## Failure modes to watch
- Through-routes with huge `stop_line_dist` must not trigger red-stop physics (`> 4000` treated as no stop).
- Roundabout-only vehicles must not preempt the main signals.
- Frame-stack size must match `STACKED_STATE_SIZE` or checkpoints fail closed (fresh weights).
