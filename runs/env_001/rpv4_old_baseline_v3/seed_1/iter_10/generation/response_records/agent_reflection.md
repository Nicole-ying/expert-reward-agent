# Response Record

Now let me write v10.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v10: Fix v9's landing_bonus exploitation, keep v5 backbone.
    - v5 progress_reward + attitude_penalty unchanged (proven backbone, score 146)
    - Replace one-shot landing_bonus (exploited via contact flicker → 98% share)
      with sustained contact_quality: small per-step reward ONLY when truly stable
    - Add gentle fuel_penalty and time_penalty for secondary efficiency objectives
    - Drop dead crash_penalty
    """
    ox, oy = obs[0], obs[1]
    ovx, ovy = obs[2], obs[3]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_av = next_obs[5]
    n_left, n_right = next_obs[6], next_obs[7]

    # ── 1. Potential-based progress (v5 backbone, unchanged) ──
    old_dist = (ox**2 + oy**2) ** 0.5
    new_dist = (nx**2 + ny**2) ** 0.5
    old_speed_sq = ovx**2 + ovy**2
    new_speed_sq = nvx**2 + nvy**2

    pot = -new_dist - 0.2 * new_speed_sq
    prev_pot = -old_dist - 0.2 * old_speed_sq
    progress_reward = 15.0 * (pot - prev_pot)

    # ── 2. Attitude penalty (v5 coefficients) ──
    attitude_penalty = 0.5 * (n_angle ** 2) + 0.15 * (n_av ** 2)

    # ── 3. Sustained contact quality (replaces exploitable one-shot landing_bonus) ──
    # Only rewards true stable resting: both legs down, near target, low speed, level
    speed = (nvx**2 + nvy**2) ** 0.5
    dist_to_target = (nx**2 + ny**2) ** 0.5
    n_both = (n_left > 0.5 and n_right > 0.5)

    contact_quality = 0.0
    if n_both and speed < 0.3 and abs(n_angle) < 0.2 and dist_to_target < 1.0:
        contact_quality = 0.3

    # ── 4. Fuel efficiency: gentle penalty for any engine use ──
    fuel_penalty = 0.03 if action != 0 else 0.0

    # ── 5. Time pressure: tiny per-step cost to incentivize faster landing ──
    time_penalty = 0.01

    total_reward = (
        progress_reward
        - attitude_penalty
        + contact_quality
        - fuel_penalty
        - time_penalty
    )

    components = {
        "progress_reward": progress_reward,
        "attitude_penalty": -attitude_penalty,
        "contact_quality": contact_quality,
        "fuel_penalty": -fuel_penalty,
        "time_penalty": -time_penalty,
    }

    return float(total_reward), components
```
