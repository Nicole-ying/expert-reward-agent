def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- progress signal: reduction in Euclidean distance to target ----
    dx_obs, dy_obs = obs[0], obs[1]
    dx_next, dy_next = next_obs[0], next_obs[1]
    dist_obs = (dx_obs ** 2 + dy_obs ** 2) ** 0.5
    dist_next = (dx_next ** 2 + dy_next ** 2) ** 0.5
    progress_delta = dist_obs - dist_next   # >0 when moving toward target

    # ---- distance-gated stability penalty ----
    # Gate: 1/(1+dist) → near 0 when far (free to maneuver),
    #                     near 1 when at target (enforce fine stability).
    # This creates a natural curriculum: aggressive far, careful near.
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]

    gate = 1.0 / (1.0 + dist_next)

    # Base weights at iter-1 levels; distance gating auto-scales them down
    # when far from target, preventing the penalty from dominating progress.
    w_vel = 0.01
    w_angle = 0.01
    w_angvel = 0.005

    stability_penalty = -gate * (
        w_vel * (abs(vx) + abs(vy))
        + w_angle * abs(angle)
        + w_angvel * abs(ang_vel)
    )

    # ---- total reward ----
    total_reward = progress_delta + stability_penalty

    components = {
        'progress_delta': progress_delta,
        'stability_penalty': stability_penalty,
        'total_reward': total_reward
    }

    return float(total_reward), components