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

    # 1. Progress — strong incentive to reduce distance, zero when hovering
    #    Replaces constant proximity bonus; only pays for actual approach.
    r_progress = 20.0 * (old_dist - new_dist)

    # 2. Stability — gentle penalty for violent motion
    r_stability = -(
        0.01 * (abs(x_vel) + abs(y_vel)) +
        0.10 * abs(body_angle) +
        0.05 * abs(ang_vel)
    )

    # 3. Landing proxy — continuous gradient toward stable touchdown
    #    Floor of 0.1 ensures signal before any leg contact, fixing 0% activation.
    proximity = max(0.0, 1.0 - new_dist / 0.5)
    stillness = max(0.0, 1.0 - (abs(x_vel) + abs(y_vel)) / 0.4)
    upright   = max(0.0, 1.0 - abs(body_angle) / 0.3)
    contact   = (left_contact + right_contact) / 2.0
    r_landing = 10.0 * proximity * stillness * upright * (0.1 + 0.9 * contact)

    total_reward = r_progress + r_stability + r_landing

    components = {
        "progress_reward": r_progress,
        "stability_penalty": r_stability,
        "landing_proxy": r_landing
    }

    return float(total_reward), components