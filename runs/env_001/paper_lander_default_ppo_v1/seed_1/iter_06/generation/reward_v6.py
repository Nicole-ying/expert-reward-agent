def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    观测:
    obs[0]: x_position_relative_to_target
    obs[1]: y_position_relative_to_pad_height
    obs[2]: x_velocity
    obs[3]: y_velocity
    obs[4]: body_angle
    obs[5]: angular_velocity
    obs[6]: left_contact_flag (0.0/1.0)
    obs[7]: right_contact_flag (0.0/1.0)
    """

    def distance(obs_arr):
        return (obs_arr[0] ** 2 + obs_arr[1] ** 2 + 1e-8) ** 0.5

    # ---- 1. Progress reward: 距离减少量 ----
    dist_old = distance(obs)
    dist_new = distance(next_obs)
    progress_reward = dist_old - dist_new

    # ---- 2. Speed tracking reward ----
    max_speed = 5.0
    d_ref = 1.0
    desired_speed = max_speed * min(dist_new / d_ref, 1.0)

    cur_speed = (next_obs[2] ** 2 + next_obs[3] ** 2 + 1e-8) ** 0.5
    speed_error = abs(cur_speed - desired_speed)
    lambda_speed = 0.2
    speed_tracking_reward = -lambda_speed * speed_error

    # ---- 3. Landing improvement: 着陆质量改善量 (state_to_improvement) ----
    proximity_threshold = 0.5
    speed_threshold = 0.25
    angle_threshold = 0.2

    def landing_quality(o):
        d = (o[0] ** 2 + o[1] ** 2 + 1e-8) ** 0.5
        s = (o[2] ** 2 + o[3] ** 2 + 1e-8) ** 0.5
        prox = max(0.0, 1.0 - d / proximity_threshold)
        slow = max(0.0, 1.0 - s / speed_threshold)
        ang = max(0.0, 1.0 - abs(o[4]) / angle_threshold)
        contact = (o[6] + o[7]) * 0.5
        return prox + slow + ang + 2.0 * contact

    Q_old = landing_quality(obs)
    Q_new = landing_quality(next_obs)

    landing_improvement_coeff = 5.0
    landing_improvement = landing_improvement_coeff * max(0.0, Q_new - Q_old)

    # ---- 总奖励 ----
    total_reward = progress_reward + speed_tracking_reward + landing_improvement

    components = {
        'progress_reward': progress_reward,
        'speed_tracking_reward': speed_tracking_reward,
        'landing_improvement': landing_improvement
    }
    return float(total_reward), components