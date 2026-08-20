# Component Analysis

## 1. Overall diagnostic
The final evaluation score is strongly negative (−42.74 mean) with a wide range [−92.66, 152.84] and 18/20 episodes truncated. This indicates that the agent has not learned a consistent landing policy – it sometimes gets a large `landing_bonus` but mostly fails to reach the pad. The current reward signal is fundamentally broken because the only meaningful driving force (`landing_bonus`) is sparse and completely dominates the total reward magnitude, while the continuous shaping signal (`shaped_progress`) is nearly zero. Per reflection rule (1), a complete rewrite of the reward framework is required, not a minor adjustment.

## 2. Component‑by‑component evaluation

### landing_bonus
- **Episode‑sum mean**: 107.0, **magnitude share**: 99%, **active rate**: 1.1%.
- This single sparse bonus (10.0 × `left` × `right`) accounts for practically all observed reward magnitude. The low active rate shows it fires only rarely (at the moment of two‑leg touch). When it does fire it gives a huge spike, but it provides no learning gradient for the rest of the flight. The agent is essentially chasing a sparse terminal reward with no continuous guidance.
- **Action**: drastically reduce its weight to non‑dominant level and enrich the contact feedback.

### shaped_progress
- **Episode‑sum mean**: 0.548, **magnitude share**: <1%, **active rate**: 99.9%.
- The progress signal `(dist_current – dist_next) / (1 + 5 |angle|)` is active almost every step but produces a vanishingly small average reward. The problem is the extremely harsh soft gate: even modest angles (e.g., 0.3 rad → gate ≈ 0.4) cut the signal heavily. Moreover, the raw progress itself may be small per step. As a result, the continuous gradient is effectively lost.
- **Action**: (a) remove or loosen the angle gate (angle should be penalised separately, not used to kill progress), (b) increase the coefficient on progress so it becomes a meaningful continuous driver.

### angular_vel_penalty
- **Episode‑sum mean**: −0.121, **magnitude share**: 0.1%, **active rate**: 99.8%.
- This penalty is negligible. It may help slightly with smoothing but is not a problem. Keep it, but the scale is irrelevant given the other changes.

## 3. Design of the new reward function
The new reward must provide **dense, balanced shaping** that guides the agent toward the centre, encourages an upright attitude, keeps velocities low, and rewards ground contact progressively (not just a single spike). The framework:

- **Progress** (coefficient 2.0) without angle gating → dense driving force toward the pad.
- **Angle penalty** (−0.5·angle²) → directly punishes tilt, replacing the old gate.
- **Angular velocity penalty** (−0.1·ang_vel², unchanged) → smoothness.
- **Linear velocity penalty** (−0.2·(vx²+vy²)) → encourages gentle, fuel‑efficient flight.
- **Contact reward** (1.0·left + 1.0·right + 3.0·left·right) → gives positive feedback for any leg contact and a larger bonus for simultaneous two‑leg contact, but with much lower magnitude than the old `landing_bonus`.

All coefficients are chosen so that no single component dominates the magnitude share in the way `landing_bonus` previously did.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next observation
    x, y, vx, vy, angle, ang_vel = next_obs[0], next_obs[1], next_obs[2], next_obs[3], next_obs[4], next_obs[5]
    left, right = next_obs[6], next_obs[7]

    # 1. Progress toward the target centre (0,0)
    prev_x, prev_y = obs[0], obs[1]
    prev_dist = (prev_x ** 2 + prev_y ** 2) ** 0.5
    curr_dist = (x ** 2 + y ** 2) ** 0.5
    progress = prev_dist - curr_dist
    progress_reward = 2.0 * progress

    # 2. Attitude penalty (keep upright)
    angle_pen = -0.5 * (angle ** 2)

    # 3. Angular velocity penalty (smooth rotation)
    angvel_pen = -0.1 * (ang_vel ** 2)

    # 4. Linear velocity penalty (gentle movement)
    vel_pen = -0.2 * (vx ** 2 + vy ** 2)

    # 5. Contact reward (any leg touch + simultaneous bonus)
    contact_reward = 1.0 * left + 1.0 * right + 3.0 * left * right

    total = progress_reward + angle_pen + angvel_pen + vel_pen + contact_reward
    components = {
        "progress": progress_reward,
        "angle_penalty": angle_pen,
        "angvel_penalty": angvel_pen,
        "vel_penalty": vel_pen,
        "contact_reward": contact_reward
    }
    return float(total), components
```