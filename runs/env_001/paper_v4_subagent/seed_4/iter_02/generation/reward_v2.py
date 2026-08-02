def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解包观测
    x_cur, y_cur = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    body_angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 超参数
    PROGRESS_WEIGHT = 2.0
    FAIL_PENALTY = -10.0
    LANDING_PROXY_WEIGHT = 1.0   # 已重新定义组件，权重内化
    ANGLE_PENALTY = 0.5
    ANG_VEL_PENALTY = 0.1

    X_BOUNDARY = 1.0
    ANGLE_CRASH = 0.8
    GROUND_Y_CLOSE = 0.2
    VEL_CRASH = 1.5
    CLOSE_THRESHOLD = 0.2         # 接近原点奖励的边界
    PROXIMITY_REWARD_FACTOR = 0.5 # 最大每步接近奖励
    SUCCESS_X_THRESH = 0.1
    SUCCESS_Y_THRESH = 0.1
    SUCCESS_VEL_THRESH = 0.2
    SUCCESS_ANGLE_THRESH = 0.1
    SUCCESS_REWARD = 100.0

    # 1. 进展信号（保持不变）
    dist_cur = (x_cur**2 + y_cur**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = PROGRESS_WEIGHT * (dist_cur - dist_next)

    # 2. 失败惩罚（保持不变）
    out_of_bounds = abs(x_next) > X_BOUNDARY
    crash = False
    if (left_contact == 1.0 or right_contact == 1.0):
        close_to_ground = y_next < GROUND_Y_CLOSE
        excessive_tilt = abs(body_angle_next) > ANGLE_CRASH
        high_impact = abs(y_vel_next) > VEL_CRASH
        if close_to_ground and (excessive_tilt or high_impact):
            crash = True
    failure = out_of_bounds or crash
    failure_penalty = FAIL_PENALTY if failure else 0.0

    # 3. 着陆引导（重构后的 landing_proxy）
    dist = (x_next**2 + y_next**2) ** 0.5
    # 弱接近奖励：在垫附近提供渐进引导
    proximity_reward = max(0.0, 1.0 - dist / CLOSE_THRESHOLD) * PROXIMITY_REWARD_FACTOR
    # 成功着陆检测
    landed = (abs(x_next) < SUCCESS_X_THRESH and
              abs(y_next) < SUCCESS_Y_THRESH and
              abs(x_vel_next) < SUCCESS_VEL_THRESH and
              abs(y_vel_next) < SUCCESS_VEL_THRESH and
              abs(body_angle_next) < SUCCESS_ANGLE_THRESH and
              (left_contact == 1.0 or right_contact == 1.0))
    success_reward = SUCCESS_REWARD if landed else 0.0
    landing_proxy_reward = LANDING_PROXY_WEIGHT * (proximity_reward + success_reward)

    # 4. 姿态稳定惩罚（保持不变）
    stability_penalty = -ANGLE_PENALTY * (body_angle_next ** 2) - ANG_VEL_PENALTY * (ang_vel_next ** 2)

    total_reward = progress + failure_penalty + landing_proxy_reward + stability_penalty

    components = {
        'progress': progress,
        'failure_penalty': failure_penalty,
        'landing_proxy': landing_proxy_reward,  # 名称不变，便于追踪
        'stability_penalty': stability_penalty
    }

    return float(total_reward), components