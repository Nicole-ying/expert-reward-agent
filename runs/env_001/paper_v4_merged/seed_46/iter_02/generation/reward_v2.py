def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    ox, oy, ovx, ovy, oangle, oav, olc, orc = obs
    nx, ny, nvx, nvy, nangle, nav, nlc, nrc = next_obs

    # Compute distances
    old_dist = (ox * ox + oy * oy) ** 0.5
    new_dist = (nx * nx + ny * ny) ** 0.5

    # --- Goal proximity progress (main learning signal) ---
    raw_progress = old_dist - new_dist
    max_delta = 5.0
    progress = max(-max_delta, min(max_delta, raw_progress))
    progress_reward = 1.0 * progress

    # --- Orientation penalty (constraint) ---
    ANGLE_THRESHOLD = 0.3
    ORIENT_WEIGHT = 0.2
    orientation_penalty = -ORIENT_WEIGHT * max(0.0, abs(nangle) - ANGLE_THRESHOLD)

    # --- Terminal success bonus (task-completion proxy) ---
    SUCCESS_DIST = 0.2
    SUCCESS_SPEED = 0.5
    SUCCESS_ANGLE = 0.2
    SUCCESS_BONUS = 0.2
    speed = (nvx * nvx + nvy * nvy) ** 0.5
    success_bonus = 0.0
    if (new_dist < SUCCESS_DIST and speed < SUCCESS_SPEED
            and abs(nangle) < SUCCESS_ANGLE
            and (nlc > 0.5 or nrc > 0.5)):
        success_bonus = SUCCESS_BONUS

    total_reward = progress_reward + orientation_penalty + success_bonus

    components = {
        "goal_proximity_progress": progress_reward,
        "orientation_penalty": orientation_penalty,
        "terminal_success_bonus": success_bonus
    }

    return float(total_reward), components