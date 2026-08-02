def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Progress reward (improvement_delta on distance to origin)
    dist_curr = (x_curr**2 + y_curr**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = 10.0 * (dist_curr - dist_next)  # positive when getting closer

    # 2. Soft landing constraints
    k = 0.5
    desired_y_vel = -k * y_next
    vertical_error = y_vel_next - desired_y_vel
    penalty_y_vel = vertical_error**2
    penalty_x_vel = x_vel_next**2
    angle_error = max(0.0, abs(angle_next) - 0.2)
    penalty_angle = angle_error**2
    ang_vel_error = max(0.0, abs(ang_vel_next) - 0.5)
    penalty_ang_vel = ang_vel_error**2

    w_y_vel = 1.0
    w_x_vel = 1.0
    w_angle = 2.0
    w_ang_vel = 0.5

    soft_landing_penalty = (w_y_vel * penalty_y_vel +
                            w_x_vel * penalty_x_vel +
                            w_angle * penalty_angle +
                            w_ang_vel * penalty_ang_vel)

    # 3. Contact-based success reward (replaces landing_approach_reward)
    # Both legs must be in contact (binary) and conditions must be safe
    contact_flag = min(left_contact, right_contact)  # 0.0 or 1.0

    # Gaussian factors for safe landing: small velocity, upright, low angular velocity
    sigma_vy = 0.2
    sigma_vx = 0.2
    sigma_angle = 0.1
    sigma_angvel = 0.2

    safe_vy = 2.718281828 ** (- (y_vel_next**2) / (sigma_vy**2))
    safe_vx = 2.718281828 ** (- (x_vel_next**2) / (sigma_vx**2))
    safe_angle = 2.718281828 ** (- (angle_next**2) / (sigma_angle**2))
    safe_angvel = 2.718281828 ** (- (ang_vel_next**2) / (sigma_angvel**2))

    contact_success_reward = 200.0 * contact_flag * safe_vy * safe_vx * safe_angle * safe_angvel

    total_reward = progress - soft_landing_penalty + contact_success_reward

    components = {
        'progress': progress,
        'soft_landing_penalty': soft_landing_penalty,
        'contact_success_reward': contact_success_reward
    }

    return float(total_reward), components