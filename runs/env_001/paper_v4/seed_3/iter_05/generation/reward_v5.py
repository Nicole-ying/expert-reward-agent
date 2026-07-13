def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current state
    x_pos, y_pos = obs[0], obs[1]
    # Next state
    nx_pos, ny_pos = next_obs[0], next_obs[1]
    nx_vel, ny_vel = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Distances to target
    prev_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (nx_pos ** 2 + ny_pos ** 2) ** 0.5

    # 1. Progress reward: positive for moving toward target, negative for moving away
    progress_weight = 5.0
    progress_reward = progress_weight * (prev_dist - next_dist)

    # 2. Velocity penalty – active only when close to target
    proximity_factor = 1.0 / (1.0 + 10.0 * next_dist)
    vel_weight = 1.0
    velocity_penalty = -vel_weight * proximity_factor * (nx_vel ** 2 + ny_vel ** 2)

    # 3. Orientation stability – keep body upright and avoid spinning
    orient_weight = 0.5
    orientation_penalty = -orient_weight * (n_angle ** 2 + 0.2 * n_angvel ** 2)

    # 4. Landing reward – staged: accessible contact gradient + quality bonus
    contact_score = left_contact + right_contact
    contact_weight = 0.3
    contact_reward = contact_weight * proximity_factor * contact_score

    vel_magnitude = (nx_vel ** 2 + ny_vel ** 2) ** 0.5
    settling_factor = 1.0 / (1.0 + 5.0 * vel_magnitude + 2.0 * abs(n_angle) + 1.0 * abs(n_angvel))
    contact_product = left_contact * right_contact
    settling_bonus_weight = 0.5
    settling_bonus = settling_bonus_weight * contact_product * proximity_factor * settling_factor

    landing_reward = contact_reward + settling_bonus

    # 5. Engine efficiency penalty – discourage unnecessary thrust
    #    action=0 (no_engine) is free; any engine use incurs a small constant cost
    engine_weight = 0.05
    engine_penalty = -engine_weight * (1.0 if action != 0 else 0.0)

    total_reward = progress_reward + velocity_penalty + orientation_penalty + landing_reward + engine_penalty

    components = {
        'progress': float(progress_reward),
        'velocity_penalty': float(velocity_penalty),
        'orientation_penalty': float(orientation_penalty),
        'landing_reward': float(landing_reward),
        'engine_penalty': float(engine_penalty)
    }

    return float(total_reward), components