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
    LANDING_PROXY_WEIGHT = 2.0
    ANGLE_PENALTY = 0.5
    ANG_VEL_PENALTY = 0.1

    X_BOUNDARY = 1.0
    ANGLE_CRASH = 0.8          # 弧度，约45度
    GROUND_Y_CLOSE = 0.2       # 接近垫面的高度
    VEL_CRASH = 1.5            # 撞击速度阈值
    DIST_LAND = 0.3            # 着陆判定距离范围
    VEL_LAND = 0.5
    ANGLE_LAND = 0.3

    # 1. 进展信号：每步距离的变化量
    dist_cur = (x_cur**2 + y_cur**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = PROGRESS_WEIGHT * (dist_cur - dist_next)   # 期望 >0

    # 2. 失败惩罚（推断终止原因）
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

    # 3. 软着陆近似信号（多条件代理）
    # 距离因子
    dist_to_pad = (x_next**2 + y_next**2) ** 0.5
    dist_factor = max(0.0, 1.0 - dist_to_pad / DIST_LAND)
    # 速度因子
    speed = abs(x_vel_next) + abs(y_vel_next)
    vel_factor = max(0.0, 1.0 - speed / VEL_LAND)
    # 姿态因子
    angle_factor = max(0.0, 1.0 - abs(body_angle_next) / ANGLE_LAND)
    # 接触因子
    contact_factor = 0.5 * (left_contact + right_contact)   # 0, 0.5, 或1

    landing_proxy = (dist_factor + vel_factor + angle_factor + contact_factor) / 4.0
    landing_proxy_reward = LANDING_PROXY_WEIGHT * landing_proxy

    # 4. 姿态/稳定轻惩罚
    stability_penalty = -ANGLE_PENALTY * (body_angle_next ** 2) - ANG_VEL_PENALTY * (ang_vel_next ** 2)

    total_reward = progress + failure_penalty + landing_proxy_reward + stability_penalty

    components = {
        'progress': progress,
        'failure_penalty': failure_penalty,
        'landing_proxy': landing_proxy_reward,
        'stability_penalty': stability_penalty
    }

    return float(total_reward), components