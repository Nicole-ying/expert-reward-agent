def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    reward_v1 for lunar-lander-like 2D flying task.
    Components:
    1. progress_reward:   -distance_to_landing_pad (dense guidance)
    2. orientation_penalty: small penalty for non‑zero body angle and angular velocity
    3. landing_bonus:       large positive reward when lander is close, slow, and touching ground
    """
    # ---- unpack observations ----
    # current state
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    # left_contact = obs[6]   # not needed for progress
    # right_contact = obs[7]

    # next state (for landing condition)
    n_x_pos = next_obs[0]
    n_y_pos = next_obs[1]
    n_x_vel = next_obs[2]
    n_y_vel = next_obs[3]
    n_body_angle = next_obs[4]
    n_angular_vel = next_obs[5]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # ---- 1. progress_reward: negative Euclidean distance to target ----
    distance_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    progress_reward = -distance_current

    # ---- 2. orientation_penalty: encourage upright and still attitude ----
    # Small coefficients so it doesn't dominate the learning signal.
    angle_coeff = 0.05
    angvel_coeff = 0.01
    orientation_penalty = -angle_coeff * abs(body_angle) - angvel_coeff * abs(angular_vel)

    # ---- 3. landing_bonus: soft proxy for successful landing ----
    # Trigger conditions:
    #   - close to the pad centre
    #   - low total speed
    #   - at least one leg in contact with ground
    close_thresh = 0.3
    speed_thresh = 0.5
    bonus_value = 10.0

    n_distance = (n_x_pos ** 2 + n_y_pos ** 2) ** 0.5
    n_speed = (n_x_vel ** 2 + n_y_vel ** 2) ** 0.5
    legs_contact = (n_left_contact > 0.5) or (n_right_contact > 0.5)

    landing_bonus = 0.0
    if n_distance < close_thresh and n_speed < speed_thresh and legs_contact:
        landing_bonus = bonus_value

    # ---- total reward ----
    total_reward = progress_reward + orientation_penalty + landing_bonus

    # components dict (only includes the terms that are summed)
    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "landing_bonus": landing_bonus
    }

    return float(total_reward), components