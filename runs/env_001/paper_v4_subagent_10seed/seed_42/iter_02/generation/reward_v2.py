def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v2: Convexified progress reward to incentivize faster approach.
    """
    # ---- Unpack observations ----
    px0, py0 = obs[0], obs[1]
    px1, py1 = next_obs[0], next_obs[1]
    vx1, vy1 = next_obs[2], next_obs[3]
    angle1  = next_obs[4]
    angvel1 = next_obs[5]
    left_leg  = next_obs[6]
    right_leg = next_obs[7]

    # ---- 1. Progress to target: convex combination of linear + quadratic ----
    dist_prev = (px0**2 + py0**2) ** 0.5
    dist_next = (px1**2 + py1**2) ** 0.5
    raw_progress = dist_prev - dist_next       # positive when approaching
    progress = max(0.0, raw_progress)           # only reward net progress
    progress_reward = progress + 2.0 * progress**2   # convex -> bigger steps get more

    # ---- 2. Orientation / stability soft constraints (unchanged) ----
    angle_penalty  = -0.01 * (angle1 ** 2)
    angvel_penalty = -0.005 * (angvel1 ** 2)
    orientation_penalty = angle_penalty + angvel_penalty

    # ---- 3. Soft landing guidance (unchanged) ----
    speed1 = (vx1**2 + vy1**2) ** 0.5
    proximity_threshold = 0.2
    if dist_next < proximity_threshold:
        contact_factor = (left_leg + right_leg) / 2.0
        speed_factor = 1.0 / (1.0 + 10.0 * speed1)
        soft_landing = contact_factor * speed_factor
    else:
        soft_landing = 0.0

    # ---- Combine components ----
    total_reward = (
        1.0 * progress_reward
        + 1.0 * orientation_penalty
        + 1.0 * soft_landing
    )

    components = {
        "progress_delta": progress_reward,
        "orientation_penalty": orientation_penalty,
        "soft_landing": soft_landing
    }
    return float(total_reward), components