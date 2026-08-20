def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    # obs: [x_pos, y_pos, x_vel, y_vel, body_angle, angular_vel, left_contact, right_contact]
    ox, oy, ovx, ovy, oangle, oav, olc, orc = obs
    nx, ny, nvx, nvy, nangle, nav, nlc, nrc = next_obs

    # Compute distances
    old_dist = (ox * ox + oy * oy) ** 0.5
    new_dist = (nx * nx + ny * ny) ** 0.5

    # --- Goal proximity progress (main learning signal) ---
    raw_progress = old_dist - new_dist   # positive when moving closer
    max_delta = 5.0                      # clip single-step changes
    progress = max(-max_delta, min(max_delta, raw_progress))
    progress_reward = 1.0 * progress     # weight = 1.0

    # --- Landing gentleness (constraint) ---
    CLOSE_DIST = 3.0
    SAFE_SPEED = 1.0
    LAND_WEIGHT = 0.5
    speed = (nvx * nvx + nvy * nvy) ** 0.5
    # Linear activation inside the close region (0 -> 1 as distance decreases)
    close_factor = max(0.0, 1.0 - new_dist / CLOSE_DIST)
    # Hinge penalty on excess speed, scaled by close_factor
    landing_penalty = -LAND_WEIGHT * max(0.0, speed - SAFE_SPEED) * close_factor

    # --- Orientation penalty (constraint) ---
    ANGLE_THRESHOLD = 0.3   # radians
    ORIENT_WEIGHT = 0.2
    orientation_penalty = -ORIENT_WEIGHT * max(0.0, abs(nangle) - ANGLE_THRESHOLD)

    # --- Terminal success bonus (task-completion proxy) ---
    SUCCESS_DIST = 0.2
    SUCCESS_SPEED = 0.5
    SUCCESS_ANGLE = 0.2
    SUCCESS_BONUS = 0.2
    success_bonus = 0.0
    if (new_dist < SUCCESS_DIST and speed < SUCCESS_SPEED
            and abs(nangle) < SUCCESS_ANGLE
            and (nlc > 0.5 or nrc > 0.5)):
        success_bonus = SUCCESS_BONUS

    total_reward = progress_reward + landing_penalty + orientation_penalty + success_bonus

    components = {
        "goal_proximity_progress": progress_reward,
        "landing_gentleness_penalty": landing_penalty,
        "orientation_penalty": orientation_penalty,
        "terminal_success_bonus": success_bonus
    }

    return float(total_reward), components