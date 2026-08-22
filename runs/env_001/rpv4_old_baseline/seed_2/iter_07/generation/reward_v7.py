def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next state
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---- Core dense signal: exponential state goodness ----
    # Encourage all state components to approach zero (landed, upright, still)
    # Higher squared penalties for y, angle, and velocities to drive descent and stability
    squared_error = (0.2 * x**2 + 1.0 * y**2 + 0.5 * vx**2 + 0.5 * vy**2 +
                     5.0 * body_angle**2 + 2.0 * angvel**2)
    state_goodness = 10.0 * (2.71828 ** (-squared_error))  # use e^(-error)
    # Maximum ~10 when fully landed, decays gracefully with any deviation

    # ---- Contact reward: encourage touching the platform ----
    contact_reward = (left_contact + right_contact) * 0.5

    # ---- Descent bonus: reward downwards progress ----
    # y decreases when moving down, so obs[1]-next_obs[1] is positive on descent
    descent_bonus = 1.0 * max(obs[1] - next_obs[1], 0.0)

    # ---- Small per‑step penalty to discourage lingering ----
    time_penalty = -0.02

    total = state_goodness + contact_reward + descent_bonus + time_penalty

    components = {
        "state_goodness": state_goodness,
        "contact_reward": contact_reward,
        "descent_bonus": descent_bonus,
        "time_penalty": time_penalty
    }
    return float(total), components