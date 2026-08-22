def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next observation
    x, y, vx, vy, angle, ang_vel = next_obs[0], next_obs[1], next_obs[2], next_obs[3], next_obs[4], next_obs[5]
    left, right = next_obs[6], next_obs[7]

    # 1. Progress toward the target centre (0,0)
    prev_x, prev_y = obs[0], obs[1]
    prev_dist = (prev_x ** 2 + prev_y ** 2) ** 0.5
    curr_dist = (x ** 2 + y ** 2) ** 0.5
    progress = prev_dist - curr_dist
    progress_reward = 2.0 * progress

    # 2. Attitude penalty (keep upright)
    angle_pen = -0.5 * (angle ** 2)

    # 3. Angular velocity penalty (smooth rotation)
    angvel_pen = -0.1 * (ang_vel ** 2)

    # 4. Linear velocity penalty (gentle movement)
    vel_pen = -0.2 * (vx ** 2 + vy ** 2)

    # 5. Contact reward (any leg touch + simultaneous bonus)
    contact_reward = 1.0 * left + 1.0 * right + 3.0 * left * right

    total = progress_reward + angle_pen + angvel_pen + vel_pen + contact_reward
    components = {
        "progress": progress_reward,
        "angle_penalty": angle_pen,
        "angvel_penalty": angvel_pen,
        "vel_penalty": vel_pen,
        "contact_reward": contact_reward
    }
    return float(total), components