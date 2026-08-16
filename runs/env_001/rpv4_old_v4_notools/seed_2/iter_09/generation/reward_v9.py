def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- distance to center (progress shaping)
    dist_prev = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = 8.0 * (dist_prev - dist_next)

    # --- velocity penalty
    vx = next_obs[2]
    vy = next_obs[3]
    velocity_penalty = -0.03 * (vx ** 2 + vy ** 2)

    # --- orientation penalty
    orientation_penalty = -0.05 * abs(next_obs[4]) - 0.03 * (next_obs[5] ** 2)

    # --- stable landing event bonus (replaces per-step contact bonus)
    # Check if any leg is touching
    leg_prev = obs[6] > 0.5 or obs[7] > 0.5
    leg_next = next_obs[6] > 0.5 or next_obs[7] > 0.5

    vel_prev = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    vel_next = (vx ** 2 + vy ** 2) ** 0.5

    angle_prev = abs(obs[4])
    angle_next = abs(next_obs[4])

    # Stable condition: touching + low speed + upright
    prev_stable = leg_prev and vel_prev < 0.2 and angle_prev < 0.1
    next_stable = leg_next and vel_next < 0.2 and angle_next < 0.1

    stable_landing_bonus = 10.0 if (next_stable and not prev_stable) else 0.0

    # --- stable contact reward: encourage maintaining stable contact
    contact_active = next_obs[6] + next_obs[7]
    stable_contact_bonus = 0.0
    if contact_active > 0.5 and abs(vy) < 0.2:
        stable_contact_bonus = 0.5  # increased to balance total reward

    # --- combine
    total_reward = (
        progress
        + velocity_penalty
        + orientation_penalty
        + stable_landing_bonus
        + stable_contact_bonus
    )

    components = {
        "progress": progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "stable_landing_bonus": stable_landing_bonus,
        "stable_contact_bonus": stable_contact_bonus,
    }

    return float(total_reward), components