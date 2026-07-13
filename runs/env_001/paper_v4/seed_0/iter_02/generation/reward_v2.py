def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x = obs[0]
    y = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    angle = obs[4]
    left_contact = obs[6]
    right_contact = obs[7]

    next_x = next_obs[0]
    next_y = next_obs[1]

    # 1. Progress: reward for reducing Euclidean distance to target pad center
    #    State-to-improvement transform: agent cannot farm by occupying a good state
    dist_old = (x ** 2 + y ** 2) ** 0.5
    dist_new = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = 5.0 * (dist_old - dist_new)

    # 2. Soft landing reward: unchanged structure, only active near the pad
    threshold_y = 0.2
    landing_weight = 2.0
    if y < threshold_y:
        contact_factor = (left_contact + right_contact) / 2.0
        speed_factor = 1.0 / (1.0 + 5.0 * (abs(x_vel) + abs(y_vel)))
        angle_factor = 1.0 / (1.0 + 10.0 * abs(angle))
        landing_bonus = landing_weight * contact_factor * speed_factor * angle_factor
        landing_reward = landing_bonus
    else:
        landing_reward = 0.0

    # 3. Fuel efficiency penalty: unchanged
    fuel_penalty = -0.05 if action != 0 else 0.0

    total_reward = progress + landing_reward + fuel_penalty

    components = {
        "progress": progress,
        "landing_reward": landing_reward,
        "fuel_penalty": fuel_penalty
    }

    return float(total_reward), components