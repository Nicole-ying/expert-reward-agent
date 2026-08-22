def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- distance to center (used for progress shaping, not for penalty)
    dist_prev = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # --- progress: strong shaping towards target
    progress = 8.0 * (dist_prev - dist_next)

    # --- velocity penalty: penalize high speed to encourage safe landing
    vx = next_obs[2]
    vy = next_obs[3]
    velocity_penalty = -0.03 * (vx ** 2 + vy ** 2)

    # --- orientation penalty: discourage large tilt
    orientation_penalty = -0.05 * abs(next_obs[4]) - 0.03 * (next_obs[5] ** 2)

    # --- contact event reward: bonus for newly touching a leg to the platform
    leg_left_prev = obs[6]
    leg_right_prev = obs[7]
    leg_left_next = next_obs[6]
    leg_right_next = next_obs[7]

    contact_event = 0.0
    if leg_left_next > 0.5 and leg_left_prev < 0.5:
        contact_event += 1.0
    if leg_right_next > 0.5 and leg_right_prev < 0.5:
        contact_event += 1.0
    contact_bonus = 5.0 * contact_event

    # --- stable contact reward: small per-step reward for maintaining contact with low vertical speed
    # Keep it small to avoid encouraging unnecessary lingering on the platform.
    contact_active = leg_left_next + leg_right_next
    stable_contact_bonus = 0.0
    if contact_active > 0.5 and abs(vy) < 0.2:
        stable_contact_bonus = 0.1   # reduced further to discourage hovering

    # --- combine
    total_reward = (
        progress
        + velocity_penalty
        + orientation_penalty
        + contact_bonus
        + stable_contact_bonus
    )

    components = {
        "progress": progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_bonus": contact_bonus,
        "stable_contact_bonus": stable_contact_bonus,
    }

    return float(total_reward), components