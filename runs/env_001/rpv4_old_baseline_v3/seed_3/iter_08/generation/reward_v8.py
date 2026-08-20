def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nxv, nyv = next_obs[2], next_obs[3]
    nangle = next_obs[4]
    nlc, nrc = next_obs[6], next_obs[7]

    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5

    # Platform detection
    prev_on_platform = (obs[6] > 0.5 and obs[7] > 0.5 and dist < 0.5)
    on_platform = (nlc > 0.5 and nrc > 0.5 and next_dist < 0.5)
    next_speed = (nxv**2 + nyv**2) ** 0.5

    # === A. Potential-based proximity: difference form prevents accumulation ===
    # Total contribution over full episode bounded to ~8 * (1 - 1/(1+init_dist))
    # Gives zero when hovering at constant distance — no more hovering exploit.
    proximity_reward = 8.0 * (1.0 / (1.0 + next_dist) - 1.0 / (1.0 + dist))

    # === B. Progress: linear distance reduction, supplementary approach signal ===
    progress_reward = 2.0 * (dist - next_dist)

    # === C. Touchdown: large one-time bonus for first platform contact ===
    touchdown_bonus = 50.0 if (not prev_on_platform and on_platform) else 0.0

    # === D. Sustained contact: strong per-step incentive to stay on platform ===
    contact_reward = 20.0 if on_platform else 0.0

    # === E. Speed penalty on platform: discourage sliding after landing ===
    speed_penalty = -5.0 * next_speed if on_platform else 0.0

    # === F. Orientation penalty: mild tilt discouragement ===
    orientation_penalty = -0.2 * abs(nangle)

    # === G. Fuel cost: light regularization for engine use ===
    fuel_cost = -0.02 if action != 0 else 0.0

    total = (proximity_reward + progress_reward + touchdown_bonus +
             contact_reward + speed_penalty + orientation_penalty + fuel_cost)

    components = {
        "proximity_reward": proximity_reward,
        "progress_reward": progress_reward,
        "touchdown_bonus": touchdown_bonus,
        "contact_reward": contact_reward,
        "speed_penalty": speed_penalty,
        "orientation_penalty": orientation_penalty,
        "fuel_cost": fuel_cost,
    }
    return float(total), components