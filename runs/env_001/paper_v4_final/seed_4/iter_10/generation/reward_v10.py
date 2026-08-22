def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解包观测
    x_next, y_next = next_obs[0], next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    body_angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---- 超参数 ----
    PROGRESS_WEIGHT = 20.0
    SURVIVAL_PENALTY = -0.08
    FAIL_PENALTY = -20.0               # 出界 / 坠毁 一次性惩罚
    STALL_GATE = 0.1                   # 远离目标且停滞时 progress 乘以此值

    ANGLE_PENALTY = 0.3
    ANG_VEL_PENALTY = 0.03

    ACTION_FUEL_PENALTY = -0.01

    # 成功着陆检测参数
    SUCCESS_DIST_THRESH = 0.15
    SUCCESS_SPEED_THRESH = 0.2
    SUCCESS_ANGLE_THRESH = 0.1
    LANDING_SUCCESS_BONUS = 150.0

    # 出界/坠毁阈值
    X_BOUNDARY = 1.0
    GROUND_Y_CLOSE = 0.15
    CRASH_ANGLE = 0.8
    CRASH_IMPACT_VEL = 1.5

    # ---- 1. 进展信号（带 stall-gate） ----
    x_cur, y_cur = obs[0], obs[1]
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5

    cur_speed = (x_vel_next ** 2 + y_vel_next ** 2) ** 0.5

    # 判断是否停滞在远离目标处（移除了角度限制）
    is_stall = (
        (dist_next > SUCCESS_DIST_THRESH) and
        (cur_speed < 0.2) and
        (left_contact < 0.5 and right_contact < 0.5)
    )
    gate = STALL_GATE if is_stall else 1.0

    progress = PROGRESS_WEIGHT * (dist_cur - dist_next) * gate

    # ---- 2. 每步存在惩罚 ----
    survival = SURVIVAL_PENALTY

    # ---- 3. 姿态/稳定惩罚 ----
    stability = -ANGLE_PENALTY * (body_angle_next ** 2) - ANG_VEL_PENALTY * (ang_vel_next ** 2)

    # ---- 4. 燃料效率惩罚 ----
    fuel = ACTION_FUEL_PENALTY if action != 0 else 0.0

    # ---- 5. 一次性成功着陆奖励 ----
    prev_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    prev_speed = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    prev_angle = abs(obs[4])
    prev_contact = max(obs[6], obs[7]) > 0.5

    cur_angle = abs(body_angle_next)
    cur_contact = max(left_contact, right_contact) > 0.5

    prev_success = (prev_dist < SUCCESS_DIST_THRESH and prev_speed < SUCCESS_SPEED_THRESH and
                    prev_angle < SUCCESS_ANGLE_THRESH and prev_contact)
    cur_success = (dist_next < SUCCESS_DIST_THRESH and cur_speed < SUCCESS_SPEED_THRESH and
                   cur_angle < SUCCESS_ANGLE_THRESH and cur_contact)

    landing_success_bonus = LANDING_SUCCESS_BONUS if (cur_success and not prev_success) else 0.0

    # ---- 6. 终止事件推断（仅出界与坠毁） ----
    out_of_bounds = abs(x_next) > X_BOUNDARY

    crash = False
    if (left_contact > 0.5 or right_contact > 0.5):
        close_to_ground = y_next < GROUND_Y_CLOSE
        excessive_tilt = abs(body_angle_next) > CRASH_ANGLE
        high_impact = abs(y_vel_next) > CRASH_IMPACT_VEL
        if close_to_ground and (excessive_tilt or high_impact):
            crash = True

    # 远离目标的停滞不再给一次性惩罚，已用 gate 处理
    failure_penalty = FAIL_PENALTY if (out_of_bounds or crash) else 0.0

    # 合并奖励
    total_reward = (progress + survival + stability + fuel +
                    landing_success_bonus + failure_penalty)

    components = {
        'progress': progress,
        'survival': survival,
        'stability': stability,
        'fuel': fuel,
        'landing_success_bonus': landing_success_bonus,
        'failure_penalty': failure_penalty
    }

    return float(total_reward), components