def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- distance to center (progress shaping) ---
    dist_prev = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = 10.0 * (dist_prev - dist_next)

    # --- velocity penalty (encourage gentle landing) ---
    vx = next_obs[2]
    vy = next_obs[3]
    velocity_penalty = -0.03 * (vx ** 2 + vy ** 2)

    # --- orientation penalty ---
    orientation_penalty = -0.05 * abs(next_obs[4]) - 0.03 * (next_obs[5] ** 2)

    # --- stable holding reward: only when perfectly stable on ground ---
    leg_contact = (next_obs[6] > 0.5) or (next_obs[7] > 0.5)
    speed = (vx ** 2 + vy ** 2) ** 0.5
    upright = abs(next_obs[4]) < 0.1
    stable_holding_bonus = 1.0 if (leg_contact and speed < 0.15 and upright) else 0.0

    # --- step penalty to discourage lingering and improve fuel efficiency ---
    step_penalty = -0.03

    # --- combine ---
    total_reward = (
        progress
        + velocity_penalty
        + orientation_penalty
        + stable_holding_bonus
        + step_penalty
    )

    components = {
        "progress": progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "stable_holding_bonus": stable_holding_bonus,
        "step_penalty": step_penalty,
    }

    return float(total_reward), components