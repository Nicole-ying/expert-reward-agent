# Response Record

Good — I have the full picture. Now let me write the improved reward function.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    obs / next_obs: 8-dim array
    [x_position, y_position, x_velocity, y_velocity, body_angle,
     angular_velocity, left_contact, right_contact]

    Design rationale:
    - Keep v5's proven potential-based progress as the approach engine
    - Add contact_reward to teach the agent to touch the pad (was missing!)
    - Add landing_bonus for the full "stable landing" configuration
    - Mild fuel_cost and crash_penalty as guardrails
    """
    # Current state
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    # Next state
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_av = next_obs[5]
    n_left = next_obs[6]
    n_right = next_obs[7]

    # ── 1. Potential-based progress (kept from v5) ──
    # Rewards moving closer AND slowing down simultaneously
    dist = (x**2 + y**2) ** 0.5
    ndist = (nx**2 + ny**2) ** 0.5
    speed_sq = vx**2 + vy**2
    nspeed_sq = nvx**2 + nvy**2

    pot = -ndist - 0.2 * nspeed_sq
    prev_pot = -dist - 0.2 * speed_sq
    progress_reward = 15.0 * (pot - prev_pot)

    # ── 2. Attitude penalty (slightly strengthened angular velocity weight) ──
    attitude_penalty = 0.5 * (n_angle ** 2) + 0.15 * (n_av ** 2)

    # ── 3. Contact reward: per-leg, dense, additive ──
    # Each leg in contact gives +0.5 per step. Easy to discover.
    # Does NOT require both legs simultaneously — builds a gradient.
    contact_reward = 0.5 * (n_left + n_right)

    # ── 4. Stable landing bonus: sparse but strong ──
    # All four success conditions met simultaneously:
    # both legs down, nearly stopped, nearly upright
    both_contacts = (n_left > 0.5 and n_right > 0.5)
    low_speed = (abs(nvx) < 0.3 and abs(nvy) < 0.3)
    upright = (abs(n_angle) < 0.2)
    landing_bonus = 3.0 if (both_contacts and low_speed and upright) else 0.0

    # ── 5. Fuel efficiency: mild penalty for any engine fire ──
    fuel_cost = 0.05 if action != 0 else 0.0

    # ── 6. Crash penalty: high-speed contact with ground ──
    contact_any = (n_left + n_right) > 0.0
    speed_mag = (nvx**2 + nvy**2) ** 0.5
    crash_penalty = speed_mag if (contact_any and speed_mag > 1.0) else 0.0

    total_reward = (
        progress_reward
        - attitude_penalty
        + contact_reward
        + landing_bonus
        - fuel_cost
        - crash_penalty
    )

    components = {
        "progress_reward": progress_reward,
        "attitude_penalty": -attitude_penalty,
        "contact_reward": contact_reward,
        "landing_bonus": landing_bonus,
        "fuel_cost": -fuel_cost,
        "crash_penalty": -crash_penalty,
    }

    return float(total_reward), components
```
