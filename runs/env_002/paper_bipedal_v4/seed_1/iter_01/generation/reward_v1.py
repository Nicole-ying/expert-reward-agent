def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # === forward progress (main learning signal) ===
    forward_vel = next_obs[2]  # horizontal_velocity
    forward_reward = 1.0 * max(0.0, forward_vel)

    # === energy penalty (secondary objective: minimize joint torques) ===
    torque_sum_sq = (
        action[0] ** 2 + action[1] ** 2 +
        action[2] ** 2 + action[3] ** 2
    )
    energy_penalty = -0.01 * torque_sum_sq

    # === upright stability (avoid falling) ===
    hull_angle = next_obs[0]
    hull_angvel = next_obs[1]
    upright_penalty = -0.5 * (hull_angle ** 2) - 0.1 * (hull_angvel ** 2)

    total_reward = forward_reward + energy_penalty + upright_penalty
    components = {
        "forward": forward_reward,
        "energy": energy_penalty,
        "upright": upright_penalty
    }
    return float(total_reward), components