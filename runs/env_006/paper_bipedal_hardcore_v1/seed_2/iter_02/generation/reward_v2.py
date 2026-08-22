def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    hull_angle = obs[0]
    hull_angvel = obs[1]
    horizontal_speed = obs[2]
    vertical_speed = obs[3]
    leg1_contact = obs[12]
    leg2_contact = obs[13]

    # 1. Forward progress (unchanged)
    forward_progress = horizontal_speed

    # 2. Balance penalty (unchanged)
    angle_threshold = 0.4
    angvel_threshold = 1.0
    angle_excess = max(0.0, abs(hull_angle) - angle_threshold)
    angvel_excess = max(0.0, abs(hull_angvel) - angvel_threshold)
    balance_penalty = -3.0 * (angle_excess ** 2) - 0.1 * (angvel_excess ** 2)

    # 3. Air-stability penalty (NEW)
    #    Punish having both feet off the ground, especially when falling downward.
    #    leg contact is 0/1, so sum 0 => both off, 1 => one on, 2 => both on.
    both_feet_off = max(0.0, 1.0 - (leg1_contact + leg2_contact))
    # Base penalty for any airborne frame (small, to allow natural brief flight)
    air_penalty = -0.3 * both_feet_off
    # Extra penalty when airborne and descending (hard landing / falling)
    vertical_fall_penalty = -1.0 * both_feet_off * max(0.0, -vertical_speed)
    air_stability_penalty = air_penalty + vertical_fall_penalty

    total_reward = forward_progress + balance_penalty + air_stability_penalty

    components = {
        'forward_progress': forward_progress,
        'balance_penalty': balance_penalty,
        'air_stability_penalty': air_stability_penalty
    }
    return float(total_reward), components