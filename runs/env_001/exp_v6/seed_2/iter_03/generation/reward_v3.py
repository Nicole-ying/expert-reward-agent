def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- position and progress delta (main learning signal, unchanged) ----
    px, py = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    d_obs = (px**2 + py**2) ** 0.5
    d_next = (nx**2 + ny**2) ** 0.5
    progress_delta = d_obs - d_next

    # ---- distance penalty (unchanged) ----
    distance_penalty = -0.005 * d_next

    # ---- stability penalty (unchanged) ----
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    stability_penalty = (
        -0.005 * abs(vx)
        - 0.005 * abs(vy)
        - 0.02 * abs(angle)
        - 0.02 * abs(ang_vel)
    )

    # ---- continuous approach bonus (REMOVED both_legs_contact binary gate) ----
    # proximity provides natural distance-gating: zero when far, gradient when close.
    # speed and angle factors give gradient toward "slow + upright" approach.
    APPROACH_DIST = 2.0      # bonus activates within this radius
    SPEED_THRESH = 2.0       # combined |vx|+|vy| threshold
    ANGLE_THRESH = 0.5       # tilt threshold

    proximity = max(0.0, 1.0 - d_next / APPROACH_DIST)
    speed_ok = max(0.0, 1.0 - (abs(vx) + abs(vy)) / SPEED_THRESH)
    angle_ok = max(0.0, 1.0 - abs(angle) / ANGLE_THRESH)

    # Product of 3 bounded [0,1] factors, scaled so a good approach gives
    # meaningful reward.  No binary gate — gradient flows at all distances < APPROACH_DIST.
    approach_bonus = proximity * speed_ok * angle_ok * 5.0

    # ---- total ----
    total = progress_delta + distance_penalty + stability_penalty + approach_bonus

    components = {
        "progress_delta_reward": progress_delta,
        "distance_penalty": distance_penalty,
        "stability_penalty": stability_penalty,
        "approach_bonus": approach_bonus,
        "total_reward": total,
    }
    return float(total), components