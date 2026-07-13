def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function with progress-based guidance toward target.
    Components:
    - progress: rewards reducing distance to target center (0,0)
    - velocity_penalty: speed penalty gated by proximity to target
    - orientation_penalty: penalty for tilt and angular velocity
    - contact_bonus: small reward for simultaneous two-leg contact, scaled by proximity
    """
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

    # 4. Dual leg contact – small bonus for proper landing stance, gated by proximity
    contact_weight = 0.5
    contact_product = left_contact * right_contact
    contact_bonus = contact_weight * contact_product * proximity_factor

    total_reward = progress_reward + velocity_penalty + orientation_penalty + contact_bonus

    components = {
        'progress': float(progress_reward),
        'velocity_penalty': float(velocity_penalty),
        'orientation_penalty': float(orientation_penalty),
        'contact_bonus': float(contact_bonus)
    }

    return float(total_reward), components