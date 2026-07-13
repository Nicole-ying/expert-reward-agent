def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current and next positions
    cx, cy = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    old_dist = (cx ** 2 + cy ** 2) ** 0.5
    new_dist = (nx ** 2 + ny ** 2) ** 0.5

    # 1. Progress: reward each step toward target, penalize retreat
    r_progress = 2.0 * (old_dist - new_dist)

    # 2. Proximity: mild always-on gradient toward (0,0)
    r_proximity = 0.5 / (1.0 + new_dist)

    # 3. Stability penalty: discourage violent motion
    r_stability = -(
        0.01 * (abs(nvx) + abs(nvy)) +
        0.1 * abs(body_angle) +
        0.05 * abs(ang_vel)
    )

    # 4. Soft landing proxy: wide thresholds, contact amplifies but never gates
    proximity = max(0.0, 1.0 - new_dist / 0.8)
    stillness = max(0.0, 1.0 - (abs(nvx) + abs(nvy)) / 0.6)
    upright   = max(0.0, 1.0 - abs(body_angle) / 0.5)
    contact_sum = left_contact + right_contact  # 0.0, 1.0, or 2.0

    r_landing = 15.0 * proximity * stillness * upright * (1.0 + contact_sum)

    total_reward = r_progress + r_proximity + r_stability + r_landing

    components = {
        "progress_reward": r_progress,
        "proximity_reward": r_proximity,
        "stability_penalty": r_stability,
        "landing_proxy": r_landing
    }

    return float(total_reward), components