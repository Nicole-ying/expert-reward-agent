def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Proximity reward (convexified) + attitude penalty + soft landing gate.
    Changed proximity_reward to 1/(1+dist^2) to strengthen gradient near target.
    """

    # ── Unpack observations ──────────────────────────────────────────
    px1, py1 = next_obs[0], next_obs[1]  # position
    vx1, vy1 = next_obs[2], next_obs[3]  # velocity
    angle1  = next_obs[4]                # body angle
    angvel1 = next_obs[5]                # angular velocity
    left_leg  = next_obs[6]              # left contact
    right_leg = next_obs[7]              # right contact

    # ── Derived signals ─────────────────────────────────────────────
    dist_next = (px1**2 + py1**2) ** 0.5
    speed = (vx1**2 + vy1**2) ** 0.5

    # ── 1. Proximity reward (convexified) ───────────────────────────
    #  1/(1+dist) → 1/(1+dist^2) to increase gradient at ~0.3–0.4.
    proximity_reward = 1.0 / (1.0 + dist_next**2)

    # ── 2. Attitude penalty ──────────────────────────────────────────
    angle_penalty  = -0.003 * (angle1 ** 2)
    angvel_penalty = -0.001 * (angvel1 ** 2)
    attitude_penalty = angle_penalty + angvel_penalty

    # ── 3. Soft landing guidance ─────────────────────────────────────
    proximity_threshold = 0.3
    if dist_next < proximity_threshold:
        contact_factor = (left_leg + right_leg) / 2.0
        speed_factor   = 1.0 / (1.0 + 10.0 * speed)
        angle_factor   = 1.0 / (1.0 + 5.0 * (angle1**2))
        soft_landing   = contact_factor * speed_factor * angle_factor
    else:
        soft_landing = 0.0

    # ── Combine ──────────────────────────────────────────────────────
    total_reward = (
        1.0 * proximity_reward
        + 1.0 * attitude_penalty
        + 2.0 * soft_landing
    )

    components = {
        "proximity_reward":   proximity_reward,
        "attitude_penalty":   attitude_penalty,
        "soft_landing":       soft_landing,
    }
    return float(total_reward), components