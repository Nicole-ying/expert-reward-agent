```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Previous state (needed for improvement calculation and touchdown transition)
    prev_x = obs[0]
    prev_y = obs[1]
    prev_lc = obs[6]
    prev_rc = obs[7]

    # Next state
    x = next_obs[0]
    y = next_obs[1]
    xv = next_obs[2]
    yv = next_obs[3]
    ang = next_obs[4]
    lc = next_obs[6]
    rc = next_obs[7]

    # 1. Landing improvement: reward progress toward centered, low-height state
    horizontal_proximity_prev = 1.0 / (1.0 + 5.0 * abs(prev_x))
    horizontal_proximity_next = 1.0 / (1.0 + 5.0 * abs(x))
    height_factor_prev = max(0.0, 1.0 - prev_y / 2.0)
    height_factor_next = max(0.0, 1.0 - y / 2.0)

    delta_horizontal = horizontal_proximity_next - horizontal_proximity_prev
    delta_height = height_factor_next - height_factor_prev

    landing_improvement = 10.0 * (max(0.0, delta_horizontal) + max(0.0, delta_height))

    # 2. Descent safety: penalize dangerous high-speed descent near ground
    descent_safety = 0.0
    if yv < 0.0 and y > 0.02:
        total_speed = (xv ** 2 + yv ** 2) ** 0.5
        height_urgency = 1.0 / (1.0 + y)
        descent_safety = -2.0 * total_speed * height_urgency

    # 3. Touchdown bonus: first moment both legs achieve ground contact
    prev_contacts = prev_lc + prev_rc
    next_contacts = lc + rc
    touchdown_bonus = 0.0
    if next_contacts >= 2.0 and prev_contacts < 2.0:
        speed_quality = max(0.0, 1.0 - (abs(xv) + abs(yv)))
        angle_quality = max(0.0, 1.0 - abs(ang) / 0.5)
        touchdown_bonus = 3.0 * (1.0 + speed_quality * angle_quality)

    # 4. Stable landed: ongoing reward for maintaining a successful landing state
    stable_landed = 0.0
    if lc > 0.5 and rc > 0.5 and y < 0.3:
        speed_score = max(0.0, 1.0 - (abs(xv) + abs(yv)))
        angle_score = max(0.0, 1.0 - abs(ang))
        stable_landed = speed_score * angle_score

    # 5. Fuel penalty: small cost for any engine use
    fuel_penalty = -0.05 if action != 0 else 0.0

    total_reward = landing_improvement + descent_safety + touchdown_bonus + stable_landed + fuel_penalty

    components = {
        "landing_improvement": landing_improvement,
        "descent_safety": descent_safety,
        "touchdown_bonus": touchdown_bonus,
        "stable_landed": stable_landed,
        "fuel_penalty": fuel_penalty
    }

    return (float(total_reward), components)
```