def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ----- weights & thresholds -----
    w_progress = 10.0
    w_contact = 50.0
    v_target = 0.3             # desired landing speed
    a_target = 0.2             # desired landing angle (rad)
    w_distance_penalty = 0.1
    w_fuel_penalty = 0.2

    # ----- distance progress (always active) -----
    prev_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = w_progress * (prev_dist - next_dist)

    # ----- continuous distance penalty (drives downward motion) -----
    distance_penalty = -w_distance_penalty * next_dist

    # ----- contact quality (only when both legs touch) -----
    contact = next_obs[6] * next_obs[7]
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle = abs(next_obs[4])

    q_speed = max(0.0, 1.0 - speed / v_target)
    q_angle = max(0.0, 1.0 - angle / a_target)
    quality = q_speed * q_angle
    contact_reward = w_contact * contact * quality

    # ----- fuel penalty (only for main engine) -----
    fuel_penalty = -w_fuel_penalty if action == 2 else 0.0

    # ----- total reward -----
    total_reward = progress + distance_penalty + contact_reward + fuel_penalty

    components = {
        'progress': progress,
        'distance_penalty': distance_penalty,
        'contact_reward': contact_reward,
        'fuel_penalty': fuel_penalty
    }
    return float(total_reward), components