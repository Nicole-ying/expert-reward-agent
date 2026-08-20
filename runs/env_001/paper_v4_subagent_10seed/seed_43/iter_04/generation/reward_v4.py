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

    # 1. Progress reward: positive when moving closer to origin
    dist_curr = (x_curr**2 + y_curr**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = 10.0 * (dist_curr - dist_next)

    # 2. Landing gate (soft multiplicative constraint, replaces hard penalty)
    # Gaussian factors with relaxed sigma to keep gate >= 0.3 in moderate conditions
    sigma_vy = 0.5
    sigma_vx = 0.4
    sigma_angle = 0.2
    sigma_angvel = 0.4

    safe_vy = 2.718281828 ** (- (y_vel_next**2) / (sigma_vy**2))
    safe_vx = 2.718281828 ** (- (x_vel_next**2) / (sigma_vx**2))
    safe_angle = 2.718281828 ** (- (angle_next**2) / (sigma_angle**2))
    safe_angvel = 2.718281828 ** (- (ang_vel_next**2) / (sigma_angvel**2))

    landing_gate = safe_vy * safe_vx * safe_angle * safe_angvel

    # 3. Contact-based success reward (kept unchanged for now)
    contact_flag = min(left_contact, right_contact)  # 0.0 or 1.0

    sigma_vy_success = 0.2
    sigma_vx_success = 0.2
    sigma_angle_success = 0.1
    sigma_angvel_success = 0.2

    safe_vy_success = 2.718281828 ** (- (y_vel_next**2) / (sigma_vy_success**2))
    safe_vx_success = 2.718281828 ** (- (x_vel_next**2) / (sigma_vx_success**2))
    safe_angle_success = 2.718281828 ** (- (angle_next**2) / (sigma_angle_success**2))
    safe_angvel_success = 2.718281828 ** (- (ang_vel_next**2) / (sigma_angvel_success**2))

    contact_success_reward = 200.0 * contact_flag * safe_vy_success * safe_vx_success * safe_angle_success * safe_angvel_success

    # Progress is gated by landing soft constraints; contact reward is additive
    total_reward = landing_gate * progress + contact_success_reward

    components = {
        'progress': progress,
        'landing_gate': landing_gate,
        'contact_success_reward': contact_success_reward
    }

    return float(total_reward), components