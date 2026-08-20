def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D vehicle-like trajectory optimization:
    Reach and stabilize on the central target pad with minimal engine use.
    """
    # Previous observation
    px_prev = obs[0]
    py_prev = obs[1]
    prev_distance = (px_prev**2 + py_prev**2)**0.5

    # Next observation
    px = next_obs[0]
    py = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]

    # Distance to target pad center
    next_distance = (px**2 + py**2)**0.5
    # Speed magnitude
    speed = (vx**2 + vy**2)**0.5

    # 1. Main learning signal: progress delta reward
    #    Positive when approaching target, negative when retreating.
    #    Transforms persistent state penalty into improvement signal.
    progress_delta = 3.0 * (prev_distance - next_distance)

    # 2. Stability constraint: light penalty on high speed, large angle, high angular velocity
    #    Helps agent learn to slow down and keep stable attitude near target
    stability_penalty = -0.1 * speed - 0.05 * abs(angle) - 0.05 * abs(ang_vel)

    # 3. Soft approaching proxy: reward getting close and slow simultaneously
    #    Acts as a smoothed "landing" surrogate without contact signals.
    sigma_dist = 0.2
    sigma_speed = 0.3
    nearness = 2.718281828 ** (-(next_distance**2) / (2.0 * sigma_dist**2))
    slowness = 2.718281828 ** (-(speed**2) / (2.0 * sigma_speed**2))
    soft_landing_reward = 1.0 * nearness * slowness

    # Combine components
    total_reward = progress_delta + stability_penalty + soft_landing_reward

    components = {
        'progress_delta': progress_delta,
        'stability_penalty': stability_penalty,
        'soft_landing_reward': soft_landing_reward
    }

    return float(total_reward), components