def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- position and progress delta ----
    px, py = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    d_obs = (px**2 + py**2) ** 0.5
    d_next = (nx**2 + ny**2) ** 0.5
    progress_delta = d_obs - d_next

    # ---- distance penalty (anchor, tiny weight) ----
    distance_penalty = -0.05 * d_next

    # ---- stability penalty (small weight) ----
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    stability_penalty = (
        -0.05 * abs(vx)
        - 0.05 * abs(vy)
        - 0.2 * abs(angle)
        - 0.2 * abs(ang_vel)
    )

    # ---- soft landing proxy (first double-contact landing) ----
    bonus = 0.0
    # Constants – tunable thresholds
    LANDING_DIST_THRESH = 0.5
    VEL_THRESH = 0.2
    ANGLE_THRESH = 0.1

    # New double contact?
    new_contact = (
        next_obs[6] == 1.0 and next_obs[7] == 1.0 and
        (obs[6] != 1.0 or obs[7] != 1.0)
    )
    if new_contact:
        if d_next < LANDING_DIST_THRESH:
            if abs(vx) < VEL_THRESH and abs(vy) < VEL_THRESH:
                if abs(angle) < ANGLE_THRESH:
                    bonus = 1.0

    # ---- total ----
    total = progress_delta + distance_penalty + stability_penalty + bonus

    components = {
        "progress_delta_reward": progress_delta,
        "distance_penalty": distance_penalty,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": bonus,
        "total_reward": total
    }
    return float(total), components