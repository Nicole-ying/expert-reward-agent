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
    k = 0.5  # desired vertical speed proportional to height
    desired_y_vel = -k * y_next
    vertical_error = y_vel_next - desired_y_vel
    penalty_y_vel = vertical_error**2

    penalty_x_vel = x_vel_next**2

    # Hinge penalties for angle and angular velocity
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

    # 3. Landing bonus (joint condition proxy)
    landing_bonus = 0.0
    if left_contact == 1.0 and right_contact == 1.0:
        speed = (x_vel_next**2 + y_vel_next**2) ** 0.5
        # Soft bonus: higher for low speed and upright
        landing_bonus = 20.0 / (1.0 + speed) * (1.0 / (1.0 + abs(angle_next)))

    total_reward = progress - soft_landing_penalty + landing_bonus

    components = {
        'progress': progress,
        'soft_landing_penalty': soft_landing_penalty,
        'landing_bonus': landing_bonus
    }

    return float(total_reward), components