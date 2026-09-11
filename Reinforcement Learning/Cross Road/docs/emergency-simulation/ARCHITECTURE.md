# Emergency Vehicle Simulation Architecture

## Purpose
Three priority classes (ambulance, fire, police) that request right-of-way from civilians, signals, and roundabout yielders without breaking 60 FPS physics.

## Fleet profiles

| Type | Length | Max speed | Visual | Role |
|------|--------|-----------|--------|------|
| AMBULANCE | 44 | 5.5 | White + red cross + R/B bar | Medical |
| FIRE | 62 | 5.2 | Red cab / amber deck + strobe | Heavy rescue |
| POLICE | 40 | 6.0 | Dark blue + split strobe | Fast intercept |

Spawn weights keep EMS rare in background traffic; dedicated keys/buttons force a unit: `A` / `F` / `P`.

## Right-of-way stack (highest first)
1. **Kinematics:** EMS ignores red/yellow stop-line clamps and roundabout yield locks.
2. **Signal preemption:** If an EMS unit is within `EMERGENCY_PREEMPT_DIST` of a *main* stop line on the opposing green, the controller inserts yellow then serves that axis. Cooldown 8 s.
3. **Civilian pull-over:** Any non-EMS vehicle with an EMS unit on the same route behind (`< SIREN_RADIUS`) or within `0.7 * SIREN_RADIUS` Euclidean distance brakes and offsets up to 12 px.
4. **Roundabout:** Circulating EMS still occupies the annulus, so civilians remain yielded at the entry line.

## Sensing and RL
- Feature `emergency_prox ∈ [0,1]` is the max of `1 - d / SIREN_RADIUS` over live EMS units.
- One-shot `REWARD_YIELD_EMERGENCY` when a civilian is slow and flagged `yielded_to_emergency`.
- `REWARD_BLOCK_EMERGENCY` if a civilian is still fast with an EMS hit on a forward LiDAR ray `< 50 px`.

## Safety
- EMS still participates in SAT collisions and headway (they do not drive through bodies).
- Fault attribution unchanged: moving into a stopped wreck is at-fault.
- Preemption is disabled for `node == ROUNDABOUT` so boulevard EMS does not flip the distant 4-way.

## Operator HUD
Navbar `EMS` chip counts live priority units. Sidebar spawners are labeled text (no emoji icons) with keyboard equivalents.

## Verification
- `spawn_fire()` / `spawn_police()` / `spawn_ambulance()` return a vehicle with `is_emergency is True`.
- A civilian on the same route with EMS behind increases `pull_over_offset`.
- Main-phase index changes after an EMS approach on the blocked green (after yellow).
