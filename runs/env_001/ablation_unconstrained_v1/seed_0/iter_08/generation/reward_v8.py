def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack states
    cx, cy = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    old_dist = (cx ** 2 + cy ** 2) ** 0.5
    new_dist = (nx ** 2 + ny ** 2) ** 0.5

    # 1. Symmetric progress: reward approach, penalize retreat, zero when stationary
    r_progress = 8.0 * (old_dist - new_dist)

    # 2. Weak proximity for gradient maintenance only
    r_proximity = 0.1 / (1.0 + new_dist)

    # 3. Stability penalty (restored best coefficients)
    r_stability = -(
        0.01 * (abs(nvx) + abs(nvy)) +
        0.1 * abs(body_angle) +
        0.05 * abs(ang_vel)
    )

    # 4. Strict landing proxy — tight thresholds, no floor, contact-gated
    proximity = max(0.0, 1.0 - new_dist / 0.3)
    stillness = max(0.0, 1.0 - (abs(nvx) + abs(nvy)) / 0.2)
    upright   = max(0.0, 1.0 - abs(body_angle) / 0.15)

    if left_contact > 0.5 and right_contact > 0.5:
        contact_factor = 1.0
    elif left_contact > 0.5 or right_contact > 0.5:
        contact_factor = 0.15
    else:
        contact_factor = 0.0

    r_landing = 30.0 * proximity * stillness * upright * contact_factor

    total_reward = r_progress + r_proximity + r_stability + r_landing

    components = {
        "progress_reward": r_progress,
        "proximity_reward": r_proximity,
        "stability_penalty": r_stability,
        "landing_proxy": r_landing
    }

    return float(total_reward), components