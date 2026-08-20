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

    # ========== 超参数 ==========
    PROGRESS_WEIGHT = 3.0
    FAIL_BOUNDS = -10.0
    FAIL_CRASH = -10.0

    ANGLE_PENALTY = 0.3
    ANG_VEL_PENALTY = 0.05
    ACTION_PENALTY = 0.05

    SOFT_LANDING_WEIGHT = 0.5
    SOFT_POS_THRESH = 0.3
    SOFT_VEL_THRESH = 0.5          # 绝对值之和阈值
    SOFT_ANGLE_THRESH = 0.2

    SUCCESS_REWARD = 100.0
    SUCCESS_X_THRESH = 0.1
    SUCCESS_Y_THRESH = 0.1
    SUCCESS_VEL_THRESH = 0.2
    SUCCESS_ANGLE_THRESH = 0.1

    X_BOUNDARY = 1.0
    CRASH_ANGLE = 0.5
    CRASH_VEL = 1.0
    CRASH_DIST = 0.5

    # ========== 1. 主进展信号 (improvement_delta) ==========
    dist_cur = (x_cur**2 + y_cur**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = PROGRESS_WEIGHT * (dist_cur - dist_next)

    # ========== 2. 失败惩罚（边界 + 不安全着陆）==========
    # 出界
    out_of_bounds = abs(x_next) > X_BOUNDARY
    boundary_penalty = FAIL_BOUNDS if out_of_bounds else 0.0

    # 坠毁：有脚接触 + 危险状态（倾角/速度/距离任意超标）
    crash = False
    contact = (left_contact == 1.0) or (right_contact == 1.0)
    if contact:
        if (abs(body_angle_next) > CRASH_ANGLE or
            abs(y_vel_next) > CRASH_VEL or
            abs(x_vel_next) > CRASH_VEL or
            dist_next > CRASH_DIST):
            crash = True
    crash_penalty = FAIL_CRASH if crash else 0.0

    failure_penalty = boundary_penalty + crash_penalty

    # ========== 3. 姿态稳定惩罚 ==========
    stability_penalty = (-ANGLE_PENALTY * (body_angle_next ** 2) 
                         - ANG_VEL_PENALTY * (ang_vel_next ** 2))

    # ========== 4. 动作效率惩罚（离散动作） ==========
    action_penalty = 0.0
    if action != 0:  # 非 no_engine 动作消耗燃料
        action_penalty = -ACTION_PENALTY

    # ========== 5. 软着陆塑造 (joint_condition_proxy, 连续) ==========
    pos_factor = max(0.0, 1.0 - dist_next / SOFT_POS_THRESH)
    vel_abs_sum = abs(x_vel_next) + abs(y_vel_next)
    vel_factor = max(0.0, 1.0 - vel_abs_sum / SOFT_VEL_THRESH)
    angle_factor = max(0.0, 1.0 - abs(body_angle_next) / SOFT_ANGLE_THRESH)
    soft_landing = SOFT_LANDING_WEIGHT * pos_factor * vel_factor * angle_factor

    # ========== 6. 成功着陆奖励（稀疏大额） ==========
    landed = (abs(x_next) < SUCCESS_X_THRESH and
              abs(y_next) < SUCCESS_Y_THRESH and
              abs(x_vel_next) < SUCCESS_VEL_THRESH and
              abs(y_vel_next) < SUCCESS_VEL_THRESH and
              abs(body_angle_next) < SUCCESS_ANGLE_THRESH and
              contact)
    success_reward = SUCCESS_REWARD if landed else 0.0

    # ========== 汇总 ==========
    total_reward = (progress + failure_penalty + stability_penalty +
                    action_penalty + soft_landing + success_reward)

    components = {
        'progress': progress,
        'failure_penalty': failure_penalty,
        'stability_penalty': stability_penalty,
        'action_penalty': action_penalty,
        'soft_landing': soft_landing,
        'success_reward': success_reward
    }

    return float(total_reward), components