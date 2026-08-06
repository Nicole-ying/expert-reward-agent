def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next state
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---- Descent encouragement: reward moving downward ----
    descent = max(0.0, obs[1] - next_obs[1])  # positive when going down
    descent_reward = descent * 5.0

    # ---- Proximity penalty: discourage horizontal drift ----
    proximity_penalty = -0.1 * abs(x)

    # ---- Contact stability reward: only when touching the platform ----
    contact = (left_contact + right_contact) > 0.5
    if contact:
        # max 10.0, penalized by deviations from upright, still, and zero angle
        raw_stability = 10.0 - 10.0 * body_angle**2 - 2.0 * vx**2 - 2.0 * vy**2 - 2.0 * angvel**2
        stability_reward = max(0.0, raw_stability)
    else:
        stability_reward = 0.0

    # ---- Fuel penalty: discourage unnecessary engine use ----
    fuel_penalty = -0.05 if action in [1, 2, 3] else 0.0

    # ---- Small per-step penalty to discourage lingering ----
    time_penalty = -0.01

    total = descent_reward + proximity_penalty + stability_reward + fuel_penalty + time_penalty

    components = {
        "descent_reward": descent_reward,
        "proximity_penalty": proximity_penalty,
        "stability_reward": stability_reward,
        "fuel_penalty": fuel_penalty,
        "time_penalty": time_penalty
    }
    return float(total), components