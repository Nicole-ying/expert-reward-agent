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

    # 2. Soft landing constraints (unchanged from original, but known to be problematic)
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

    # 3. Landing approach reward (continuous bounded factor — REPLACES dead landing_bonus)
    # Effective height above ground (non-negative)
    h_eff = y_next if y_next > 0.0 else 0.0

    # Desired vertical speed for current height
    v_desired = -0.5 * h_eff

    # Gaussian factors: each in (0, 1], peaking at ideal condition
    sigma_h = 0.5
    sigma_vx = 0.2
    sigma_vy = 0.3
    sigma_angle = 0.15

    factor_height = 2.718281828 ** (- (h_eff**2) / (sigma_h**2))
    factor_vx = 2.718281828 ** (- (x_vel_next**2) / (sigma_vx**2))
    factor_vy = 2.718281828 ** (- ((y_vel_next - v_desired)**2) / (sigma_vy**2))
    factor_angle = 2.718281828 ** (- (angle_next**2) / (sigma_angle**2))

    # Combined approach reward: high only when all conditions are near-ideal
    w_approach = 5.0
    landing_approach_reward = w_approach * factor_height * factor_vx * factor_vy * factor_angle

    total_reward = progress - soft_landing_penalty + landing_approach_reward

    components = {
        'progress': progress,
        'soft_landing_penalty': soft_landing_penalty,
        'landing_approach_reward': landing_approach_reward
    }

    return float(total_reward), components