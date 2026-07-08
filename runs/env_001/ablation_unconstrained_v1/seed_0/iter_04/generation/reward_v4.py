def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- Previous state ---
    old_x = obs[0]
    old_y = obs[1]
    old_dist = (old_x**2 + old_y**2)**0.5

    # --- Next state ---
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    new_dist = (x_pos**2 + y_pos**2)**0.5

    # 1. Progress delta — reward approaching, penalize retreating, zero when static
    r_progress = 5.0 * (old_dist - new_dist)

    # 2. Small proximity bonus — faint attractor near target (prevents gradient collapse)
    r_proximity = 0.2 / (1.0 + new_dist)

    # 3. Stability penalty — discourage violent motion
    r_stability = -(
        0.02 * (abs(x_vel) + abs(y_vel)) +
        0.15 * abs(body_angle) +
        0.08 * abs(ang_vel)
    )

    # 4. Landing proxy — only pays with leg contact (no 0.1 floor)
    proximity = max(0.0, 1.0 - new_dist / 0.5)
    stillness = max(0.0, 1.0 - (abs(x_vel) + abs(y_vel)) / 0.4)
    upright   = max(0.0, 1.0 - abs(body_angle) / 0.3)
    contact_avg = (left_contact + right_contact) / 2.0
    r_landing = 12.0 * proximity * stillness * upright * contact_avg

    # 5. Time pressure — constant gradient toward faster completion
    r_time = -0.003

    total_reward = r_progress + r_proximity + r_stability + r_landing + r_time

    components = {
        "progress_reward": r_progress,
        "proximity_bonus": r_proximity,
        "stability_penalty": r_stability,
        "landing_proxy": r_landing,
        "time_penalty": r_time
    }

    return float(total_reward), components