def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x, y = obs[0], obs[1]
    xv, yv = obs[2], obs[3]
    angle = obs[4]
    # angv = obs[5]  # not used directly, could be added later if needed
    lc, rc = obs[6], obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nxv, nyv = next_obs[2], next_obs[3]
    nangle = next_obs[4]
    # nangv = next_obs[5]
    nlc, nrc = next_obs[6], next_obs[7]

    # Euclidean distances
    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5

    # ================= A. Progress: delta distance =================
    # Dense reward for moving toward the target platform (0,0)
    w_progress = 10.0
    progress_reward = w_progress * (dist - next_dist)

    # ================= B. Velocity penalty near target =================
    # Apply only when close to the target (dist < d_threshold)
    d_threshold = 0.5  # meters, safe distance to start slowing down
    proximity_factor = max(0.0, 1.0 - dist / d_threshold)  # 1 at target, 0 beyond threshold

    safe_vx = 0.2
    safe_vy = 0.3
    excess_vx = max(0.0, abs(xv) - safe_vx)
    excess_vy = max(0.0, abs(yv) - safe_vy)
    w_vel_penalty = 2.0
    velocity_penalty = -w_vel_penalty * proximity_factor * (excess_vx + excess_vy)

    # ================= C. Touchdown event reward =================
    touchdown_reward = 0.0
    prev_double_contact = (lc > 0.5 and rc > 0.5)
    next_double_contact = (nlc > 0.5 and nrc > 0.5)
    if not prev_double_contact and next_double_contact:
        # Check soft landing requirements on the touchdown step
        speed_ok = abs(nxv) < safe_vx and abs(nyv) < safe_vy
        angle_ok = abs(nangle) < 0.2  # radians, near upright
        if speed_ok and angle_ok:
            touchdown_reward = 50.0

    total_reward = progress_reward + velocity_penalty + touchdown_reward

    components = {
        "progress_reward": progress_reward,
        "velocity_penalty": velocity_penalty,
        "touchdown_reward": touchdown_reward
    }
    return float(total_reward), components