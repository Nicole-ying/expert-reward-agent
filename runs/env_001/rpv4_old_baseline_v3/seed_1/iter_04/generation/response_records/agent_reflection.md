# Response Record

Now let me write the v4 reward. This restores v1's proven skeleton (30x equal-weight proximity, same attitude coefficients, same velocity/angle factors for landing) with three targeted enhancements: a progress-based descent bonus, 3x stronger landing reward, and mild fuel efficiency pressure.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v4: Restore v1's proven skeleton + descent incentive + stronger landing + fuel cost.

    Root cause of v3 failure: altitude_reward was a state-based per-step reward
    (0.1/(1+3y)) that created a "hover forever" local optimum. Agent learned to stay
    at moderate altitude collecting ~0.03/step rather than risk landing.

    Fixes:
    - REMOVE altitude_reward, partial_contact, ground_speed_penalty (all dead/harmful)
    - RESTORE v1's 30x equal-weight Euclidean proximity_progress
    - RESTORE v1's attitude_penalty coefficients and soft_landing velocity/angle factors
    - ADD descent_bonus: progress-based (NOT state-based) vertical incentive
    - STRENGTHEN soft_landing: 15x (3x v1's 5x)
    - ADD fuel_cost: mild penalty for engine use
    """
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    old_x, old_y = obs[0], obs[1]

    # ── 1. proximity_progress: v1's proven 30x equal-weight Euclidean delta ──
    old_dist = (old_x ** 2 + old_y ** 2) ** 0.5
    new_dist = (x ** 2 + y ** 2) ** 0.5
    proximity_progress = 30.0 * (old_dist - new_dist)

    # ── 2. descent_bonus: progress-based vertical incentive ──
    # CRITICAL: this rewards actual downward *movement*, not static low altitude.
    # Hovering gives zero. Ascending gives zero. No local optimum to exploit.
    dy = old_y - y
    descent_bonus = 5.0 * max(0.0, dy)

    # ── 3. soft_landing: enhanced landing reward (3x v1's 5.0) ──
    contact_score = left_contact * right_contact
    vx_abs = abs(vx)
    vy_abs = abs(vy)
    velocity_factor = 1.0 / (1.0 + 10.0 * (vx_abs + vy_abs))
    angle_factor = 1.0 / (1.0 + 5.0 * abs(angle))
    soft_landing = 15.0 * contact_score * velocity_factor * angle_factor

    # ── 4. attitude_penalty: v1's exact coefficients ──
    attitude_penalty = -0.5 * (angle ** 2) - 0.1 * (ang_vel ** 2)

    # ── 5. fuel_cost: mild penalty for any engine use ──
    fuel_cost = -0.1 if action != 0 else 0.0

    total = (
        proximity_progress
        + descent_bonus
        + soft_landing
        + attitude_penalty
        + fuel_cost
    )

    components = {
        "proximity_progress": proximity_progress,
        "descent_bonus": descent_bonus,
        "soft_landing": soft_landing,
        "attitude_penalty": attitude_penalty,
        "fuel_cost": fuel_cost,
    }
    return float(total), components
```
