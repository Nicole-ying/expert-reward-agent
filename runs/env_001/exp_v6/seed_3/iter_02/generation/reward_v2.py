def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- main progress signal: reduction in Euclidean distance to target ----
    # target position is (0, 0) in the relative coordinate system
    dx_obs, dy_obs = obs[0], obs[1]
    dx_next, dy_next = next_obs[0], next_obs[1]
    dist_obs = (dx_obs ** 2 + dy_obs ** 2) ** 0.5
    dist_next = (dx_next ** 2 + dy_next ** 2) ** 0.5
    progress_delta = dist_obs - dist_next   # positive when moving closer

    # ---- light stability penalty on next observation ----
    # penalise linear velocity, body angle and angular velocity
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]

    # weights reduced 10x from previous iteration (were 0.01, 0.01, 0.005)
    # previous ratio_to_progress was -0.85 — penalty dominated progress entirely
    w_vel = 0.001
    w_angle = 0.001
    w_angvel = 0.0005

    stability_penalty = (
        - w_vel * (abs(vx) + abs(vy))
        - w_angle * abs(angle)
        - w_angvel * abs(ang_vel)
    )

    # ---- total reward ----
    total_reward = progress_delta + stability_penalty

    # ---- component logging ----
    components = {
        'progress_delta': progress_delta,
        'stability_penalty': stability_penalty,
        'total_reward': total_reward
    }

    return float(total_reward), components