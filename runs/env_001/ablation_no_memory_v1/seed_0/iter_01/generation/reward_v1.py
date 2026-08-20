def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Compute Euclidean distance to target (landing pad center)
    d_curr = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    d_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # Main learning signal: progress towards the target
    progress = d_curr - d_next   # positive when moving closer
    # Clip progress to avoid huge single-step rewards (safe range for 2D normalised-like coordinates)
    progress_clipped = max(-0.5, min(0.5, progress))
    progress_reward = 10.0 * progress_clipped

    # Stability penalty: discourage high speed, large angle and angular velocity
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]

    # Vertical velocity is penalised less to allow gentle descent
    stability_penalty = -0.5 * abs(vx) - 0.2 * abs(vy) - 0.5 * abs(angle) - 0.1 * abs(angular_vel)

    # Soft landing proxy: dense reward when very close, slow, upright and both legs in contact
    dist_thresh = 0.05
    vel_thresh = 0.1
    angle_thresh = 0.05
    angvel_thresh = 0.05
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    if (d_next < dist_thresh and
        abs(vx) < vel_thresh and
        abs(vy) < vel_thresh and
        abs(angle) < angle_thresh and
        abs(angular_vel) < angvel_thresh and
        left_contact > 0.5 and
        right_contact > 0.5):
        landing_bonus = 1.0
    else:
        landing_bonus = 0.0

    total_reward = progress_reward + stability_penalty + landing_bonus

    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'landing_bonus': landing_bonus
    }
    return float(total_reward), components