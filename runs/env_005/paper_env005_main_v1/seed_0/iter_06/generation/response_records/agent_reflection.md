# Response Record

## 1. `evidence`
Current score=11.05 (first positive, best so far) but 19/20 episodes terminated early from body_height_outside_healthy_range, mean episode length only 252 steps, and no existing component uses body_z (obs[0]) despite it being the dominant failure mode.

## 2. `behavior_diagnosis`
The agent moves forward aggressively (forward_velocity_reward=247.5, 67.6% magnitude share) but collapses rapidly — it trades survival for speed because nothing in the reward penalizes height decay toward the 0.2 termination boundary.

## 3. `signal_completeness`
Missing a critical signal: body_z proximity to the healthy-range boundaries. The orientation_reward addresses tilt but not vertical collapse; the agent can be upright while sinking. A height-deviation penalty is both necessary and reachable via obs[0].

## 4. `selected_level`
**Level 2** — a necessary signal (height management) is entirely absent, and the termination evidence (19/20 height-violation failures) directly supports adding it. This is the `unbounded_to_bounded` pattern applied to body_z: introduce a bounded penalty that grows as body_z approaches the unhealthy thresholds.

## 5. `selected_intervention`
Add one new component: `height_deviation_penalty`. It penalizes squared deviation of body_z from 0.6 (center of [0.2, 1.0]), normalized by half-range 0.4, with coefficient 1.0. All existing components and weights are preserved unchanged.

## 6. `falsifiable_hypothesis`
Giving the agent a differentiable penalty for approaching the height boundaries should cause it to learn a gait that maintains healthy body_z, increasing survival time and thereby accumulating more forward reward — raising total score substantially above 11.05.

## 7. `expected_next_round`
- `terminated` should drop below 19/20 (fewer height-violation failures)
- `episode_length` should increase beyond ~252
- `score` should improve as longer survival accumulates forward velocity reward
- `height_deviation_penalty` should show negative episode_sum_mean with active_rate=100%

## 8. `main_risk`
If the penalty is too strong, the agent may freeze or creep to minimize height deviation, trading all forward progress for safety — resulting in long episodes but low forward velocity and stagnant score.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Materially different reward design (v2) — adds height deviation penalty
    to prevent body collapse, which caused 19/20 early terminations in v1.
    All v1 components and weights are preserved unchanged.
    """
    # next_obs indices
    BODY_X_VEL = 13
    BODY_Y_VEL = 14
    ROLL_VEL = 16
    PITCH_VEL = 17

    # Quaternion components: assumed order (w, x, y, z)
    qw = next_obs[1]
    qx = next_obs[2]
    qy = next_obs[3]
    qz = next_obs[4]
    body_up_z = 1.0 - 2.0 * (qx * qx + qy * qy)

    # Body height (vertical position)
    body_z = next_obs[0]

    forward_vel = next_obs[BODY_X_VEL]
    lateral_vel = next_obs[BODY_Y_VEL]
    roll_vel = next_obs[ROLL_VEL]
    pitch_vel = next_obs[PITCH_VEL]

    # Weights — v1 weights preserved
    w_fwd    = 1.0
    w_orient = 0.3
    w_lat    = 0.2
    w_ang    = 0.05

    # New height deviation penalty
    # Healthy range: [0.2, 1.0], center = 0.6, half-range = 0.4
    # At body_z=0.2 or 1.0, penalty = -1.0; at center 0.6, penalty = 0
    w_height = 1.0
    height_deviation = (body_z - 0.6) / 0.4
    height_deviation_penalty = -w_height * (height_deviation ** 2)

    # Components (v1 preserved)
    forward_reward = w_fwd * forward_vel
    orientation_reward = w_orient * body_up_z
    lateral_penalty = -w_lat * abs(lateral_vel)
    angular_penalty = -w_ang * (abs(roll_vel) + abs(pitch_vel))

    total_reward = (
        forward_reward
        + orientation_reward
        + lateral_penalty
        + angular_penalty
        + height_deviation_penalty
    )

    components = {
        "forward_velocity_reward": forward_reward,
        "orientation_reward": orientation_reward,
        "lateral_velocity_penalty": lateral_penalty,
        "angular_velocity_penalty": angular_penalty,
        "height_deviation_penalty": height_deviation_penalty,
    }
    return float(total_reward), components
```
