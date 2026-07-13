def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    cx, cy = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    old_dist = (cx ** 2 + cy ** 2) ** 0.5
    new_dist = (nx ** 2 + ny ** 2) ** 0.5

    # 1. Potential-based progress: only rewards movement toward target
    #    potential = 1/(1+d) is bounded in [0,1], gradient strongest near target
    old_potential = 1.0 / (1.0 + old_dist)
    new_potential = 1.0 / (1.0 + new_dist)
    r_progress = 15.0 * (new_potential - old_potential)

    # 2. Stability penalty: discourage violent motion
    r_stability = -(
        0.01 * (abs(nvx) + abs(nvy)) +
        0.1 * abs(body_angle) +
        0.05 * abs(ang_vel)
    )

    # 3. Soft landing proxy: wide thresholds to guarantee early gradient
    #    Each factor is max(0, 1 - value/threshold), activates gradually
    proximity = max(0.0, 1.0 - new_dist / 1.5)
    stillness = max(0.0, 1.0 - (abs(nvx) + abs(nvy)) / 1.2)
    upright   = max(0.0, 1.0 - abs(body_angle) / 1.0)
    contact   = (left_contact + right_contact) / 2.0
    r_landing = 10.0 * proximity * stillness * upright * (0.1 + 0.9 * contact)

    total_reward = r_progress + r_stability + r_landing

    components = {
        "progress_reward": r_progress,
        "stability_penalty": r_stability,
        "landing_proxy": r_landing
    }

    return float(total_reward), components