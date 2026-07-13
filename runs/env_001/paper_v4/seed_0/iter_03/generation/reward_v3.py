def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next_obs for state-after-action evaluation
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_angle = next_obs[4]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]

    # 1. Proximity: state-based bounded reward for being near target center
    dist = (next_x ** 2 + next_y ** 2) ** 0.5
    proximity = 0.1 / (1.0 + dist)

    # 2. Landing reward: product_to_noncollapsing_joint transform
    #    Old: contact_factor * speed_factor * angle_factor (collapses to ~0)
    #    New: additive independent contributions, gated by height
    threshold_y = 0.5
    if next_y < threshold_y:
        # Contact: each leg independently contributes, range [0, 2]
        contact_score = next_left_contact + next_right_contact
        # Speed: bounded linear decay, 1 at rest, 0 when total speed >= 2.0
        total_speed = abs(next_x_vel) + abs(next_y_vel)
        speed_score = max(0.0, 1.0 - total_speed / 2.0)
        # Angle: bounded linear decay, 1 when upright, 0 when |angle| >= 0.5
        angle_score = max(0.0, 1.0 - abs(next_angle) / 0.5)
        landing_reward = 1.0 * (contact_score + speed_score + angle_score)
    else:
        landing_reward = 0.0

    # 3. Fuel efficiency penalty
    fuel_penalty = -0.05 if action != 0 else 0.0

    total_reward = proximity + landing_reward + fuel_penalty

    components = {
        "proximity": proximity,
        "landing_reward": landing_reward,
        "fuel_penalty": fuel_penalty
    }

    return float(total_reward), components