def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── Unpack observations ─────────────────────────────────────────
    px_prev, py_prev = obs[0], obs[1]          # previous position
    px, py = next_obs[0], next_obs[1]          # current position
    vx, vy = next_obs[2], next_obs[3]          # velocity
    angle  = next_obs[4]                       # body angle
    angvel = next_obs[5]                       # angular velocity
    left_leg  = next_obs[6]                    # left contact
    right_leg = next_obs[7]                    # right contact

    # ── Derived signals ─────────────────────────────────────────────
    dist_prev = (px_prev**2 + py_prev**2) ** 0.5
    dist_next = (px**2 + py**2) ** 0.5
    speed     = (vx**2 + vy**2) ** 0.5

    # ── 1. Proximity reward (static distance attractor) ─────────────
    proximity_reward = 1.0 / (1.0 + dist_next**2)

    # ── 2. Attitude penalty (unchanged) ─────────────────────────────
    attitude_penalty = -0.003 * (angle ** 2) - 0.001 * (angvel ** 2)

    # ── 3. Progress reward (NEW) ────────────────────────────────────
    # Reward each step that reduces distance to origin.
    delta_dist = dist_prev - dist_next          # positive when approaching
    progress_reward = 0.5 * max(0.0, delta_dist)

    # ── 4. Soft landing (gate kept, activates below 0.3) ────────────
    if dist_next < 0.3:
        contact_factor = (left_leg + right_leg) / 2.0
        speed_factor   = 1.0 / (1.0 + 10.0 * speed)
        angle_factor   = 1.0 / (1.0 + 5.0 * (angle ** 2))
        soft_landing   = contact_factor * speed_factor * angle_factor
    else:
        soft_landing = 0.0

    # ── Combine ─────────────────────────────────────────────────────
    total_reward = (
        1.0 * proximity_reward
        + 1.0 * attitude_penalty
        + 1.0 * progress_reward
        + 2.0 * soft_landing
    )

    components = {
        "proximity_reward": proximity_reward,
        "attitude_penalty": attitude_penalty,
        "progress_reward": progress_reward,
        "soft_landing": soft_landing,
    }
    return float(total_reward), components