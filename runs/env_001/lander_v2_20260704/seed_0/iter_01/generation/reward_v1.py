def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract next observations
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Main learning signal: negative distance to landing target (dense progress guiding)
    dist_to_target = (x_pos ** 2 + y_pos ** 2) ** 0.5
    distance_reward = -dist_to_target

    # 2. Stability constraint: discourage high speed, tilt, and rotation (light penalty)
    stab_weight = 0.01
    stability_penalty = -stab_weight * (abs(x_vel) + abs(y_vel) + abs(angle) + abs(ang_vel))

    # 3. Soft landing proxy: bonus when near target, low speed, upright, and both legs contact
    near_target = dist_to_target < 0.2
    low_speed = abs(x_vel) < 0.2 and abs(y_vel) < 0.2
    stable_angle = abs(angle) < 0.1
    low_ang_vel = abs(ang_vel) < 0.1
    both_legs = (left_contact == 1.0) and (right_contact == 1.0)

    if near_target and low_speed and stable_angle and low_ang_vel and both_legs:
        soft_landing_proxy = 1.0
    else:
        soft_landing_proxy = 0.0

    total_reward = distance_reward + stability_penalty + soft_landing_proxy

    components = {
        "distance_reward": distance_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy
    }

    return float(total_reward), components