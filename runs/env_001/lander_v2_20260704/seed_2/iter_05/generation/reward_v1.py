def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- 1. Progress delta: reward for reducing distance to goal ---
    dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    raw_progress = dist - dist_next
    # Clip to keep the signal bounded and avoid huge single‑step rewards
    progress = max(-2.0, min(2.0, raw_progress))

    # --- 2. Stability cost with altitude-dependent weighting ---
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    alt = next_obs[1]  # relative height above landing pad (goal y = 0)

    # The closer to the ground, the stronger the requirement for low speed / angle.
    altitude_factor = 1.0 + 2.718281828 ** (-alt / 0.5)  # e^(-2*alt), approx 1 far away, up to 2 near ground

    stability_cost = (0.2 * abs(vx) +
                      0.2 * abs(vy) +
                      0.2 * abs(angle) +
                      0.2 * abs(angvel)) * altitude_factor

    # --- Combine ---
    total_reward = 10.0 * progress - stability_cost

    components = {
        "progress_delta": 10.0 * progress,
        "weighted_stability_penalty": -stability_cost
    }
    return float(total_reward), components