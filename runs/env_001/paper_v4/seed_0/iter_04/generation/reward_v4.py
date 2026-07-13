def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Previous state
    prev_y = obs[1]
    prev_x_vel = obs[2]
    prev_y_vel = obs[3]
    prev_angle = obs[4]
    prev_left_contact = obs[6]
    prev_right_contact = obs[7]

    # Next state
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_angle = next_obs[4]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]

    # 1. Proximity: unchanged state-based bounded reward for being near target center
    dist = (next_x ** 2 + next_y ** 2) ** 0.5
    proximity = 0.1 / (1.0 + dist)

    # 2. Landing reward: state_to_improvement transform
    #    old: persistent absolute landing quality -> farmed by hovering
    #    new: potential difference rewards improvement, zero for maintaining
    def landing_potential(y, left_c, right_c, xv, yv, angle, threshold=0.5):
        if y < threshold:
            contact_score = left_c + right_c
            total_speed = abs(xv) + abs(yv)
            speed_score = max(0.0, 1.0 - total_speed / 2.0)
            angle_score = max(0.0, 1.0 - abs(angle) / 0.5)
            return contact_score + speed_score + angle_score
        return 0.0

    prev_pot = landing_potential(prev_y, prev_left_contact, prev_right_contact,
                                 prev_x_vel, prev_y_vel, prev_angle)
    curr_pot = landing_potential(next_y, next_left_contact, next_right_contact,
                                 next_x_vel, next_y_vel, next_angle)

    landing_reward = 2.0 * (curr_pot - prev_pot)

    # 3. Fuel efficiency penalty: unchanged
    fuel_penalty = -0.05 if action != 0 else 0.0

    total_reward = proximity + landing_reward + fuel_penalty

    components = {
        "proximity": proximity,
        "landing_reward": landing_reward,
        "fuel_penalty": fuel_penalty
    }

    return float(total_reward), components