def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward for 2D lunar-lander-style goal reaching task.
    """
    # ---- Unpack observations ----
    # obs: current state; next_obs: post-action state
    px0, py0 = obs[0], obs[1]
    px1, py1 = next_obs[0], next_obs[1]
    vx1, vy1 = next_obs[2], next_obs[3]
    angle1  = next_obs[4]
    angvel1 = next_obs[5]
    left_leg  = next_obs[6]
    right_leg = next_obs[7]

    # ---- 1. Progress to target: delta in Euclidean distance to (0,0) ----
    dist_prev = (px0**2 + py0**2) ** 0.5
    dist_next = (px1**2 + py1**2) ** 0.5
    progress_delta = dist_prev - dist_next   # positive when approaching

    # ---- 2. Orientation / stability soft constraints ----
    # Penalize large tilt and high angular velocity (use next_obs state)
    angle_penalty    = -0.01 * (angle1 ** 2)
    angvel_penalty   = -0.005 * (angvel1 ** 2)
    orientation_penalty = angle_penalty + angvel_penalty

    # ---- 3. Soft landing guidance (proximity-triggered proxy) ----
    # Activates only when the agent is close to the target pad.
    speed1 = (vx1**2 + vy1**2) ** 0.5
    proximity_threshold = 0.2          # tuned for the environment scale
    if dist_next < proximity_threshold:
        # contact factor: average of left/right leg contact (0..1)
        contact_factor = (left_leg + right_leg) / 2.0
        # speed smooth factor: 1 when speed=0, decays with higher speed
        speed_factor = 1.0 / (1.0 + 10.0 * speed1)
        soft_landing = contact_factor * speed_factor
    else:
        soft_landing = 0.0

    # ---- Combine components ----
    total_reward = (
        1.0 * progress_delta
        + 1.0 * orientation_penalty
        + 1.0 * soft_landing
    )

    components = {
        "progress_delta": progress_delta,
        "orientation_penalty": orientation_penalty,
        "soft_landing": soft_landing
    }
    return float(total_reward), components