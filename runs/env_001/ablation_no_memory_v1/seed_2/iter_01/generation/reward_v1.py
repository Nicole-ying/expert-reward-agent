def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lander reaching a central target pad.
    
    obs: [x, y, vx, vy, angle, angular_vel, left_contact, right_contact]
    next_obs: same structure after taking action
    """
    # --- Unpack observations ---
    x, y = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    omega_next = next_obs[5]
    left_contact_next, right_contact_next = next_obs[6], next_obs[7]

    # --- 1. Main learning signal: progress towards the target pad ---
    dist_current = (x**2 + y**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress_reward = dist_current - dist_next   # positive if getting closer

    # --- 2. Light stability penalty: suppress extreme velocity / angle / angular velocity ---
    speed_next = (vx_next**2 + vy_next**2) ** 0.5
    stability_penalty = -0.01 * speed_next - 0.02 * abs(angle_next) - 0.02 * abs(omega_next)

    # --- 3. Soft landing proxy: bonus when close to pad with good conditions ---
    # Conditions: near target, low speed, upright, both legs in contact
    near_target = abs(x_next) < 0.15 and abs(y_next) < 0.3
    low_speed = speed_next < 0.5
    upright = abs(angle_next) < 0.3
    both_contact = (left_contact_next == 1.0) and (right_contact_next == 1.0)

    landing_bonus = 0.5 if (near_target and low_speed and upright and both_contact) else 0.0

    # --- Total reward ---
    total_reward = progress_reward + stability_penalty + landing_bonus

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus
    }

    return float(total_reward), components