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
    SUCCESS_BONUS = 200.0
    FAIL_PENALTY = -30.0               # 出界/坠毁一次性惩罚

    ANGLE_PENALTY = 0.3
    ANG_VEL_PENALTY = 0.03

    ACTION_FUEL_PENALTY = -0.01

    # 成功阈值
    LAND_DIST_THRESHOLD = 0.2
    LAND_SPEED_THRESHOLD = 0.2
    LAND_ANGLE_THRESHOLD = 0.15
    LAND_CONTACT_REQUIRED = True

    # 出界/坠毁阈值
    X_BOUNDARY = 1.0
    GROUND_Y_CLOSE = 0.15
    CRASH_ANGLE = 0.8
    CRASH_IMPACT_VEL = 1.5

    # ---- 1. 进展信号 ----
    x_cur, y_cur = obs[0], obs[1]
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    progress = PROGRESS_WEIGHT * (dist_cur - dist_next)

    # ---- 2. 每步存在惩罚 ----
    survival = SURVIVAL_PENALTY

    # ---- 3. 姿态/稳定惩罚 ----
    stability = -ANGLE_PENALTY * (body_angle_next ** 2) - ANG_VEL_PENALTY * (ang_vel_next ** 2)

    # ---- 4. 燃料效率惩罚 ----
    fuel = ACTION_FUEL_PENALTY if action != 0 else 0.0

    # ---- 5. 终止事件推断 ----
    # 成功着陆条件
    dist_to_pad = dist_next
    speed = (x_vel_next ** 2 + y_vel_next ** 2) ** 0.5
    angle_ok = abs(body_angle_next) < LAND_ANGLE_THRESHOLD
    contact_ok = (left_contact > 0.5) or (right_contact > 0.5)
    success = (dist_to_pad < LAND_DIST_THRESHOLD and
               speed < LAND_SPEED_THRESHOLD and
               angle_ok and
               contact_ok)

    # 出界
    out_of_bounds = abs(x_next) > X_BOUNDARY

    # 坠毁（原有逻辑）
    crash = False
    if (left_contact > 0.5 or right_contact > 0.5):
        close_to_ground = y_next < GROUND_Y_CLOSE
        excessive_tilt = abs(body_angle_next) > CRASH_ANGLE
        high_impact = abs(y_vel_next) > CRASH_IMPACT_VEL
        if close_to_ground and (excessive_tilt or high_impact):
            crash = True

    # 出界或坠毁的一次性惩罚（移除 early_stop 连续惩罚）
    failure_penalty = FAIL_PENALTY if (out_of_bounds or crash) else 0.0

    # 成功奖励
    landing_bonus = SUCCESS_BONUS if success else 0.0

    # 合并
    total_reward = (progress + survival + stability + fuel +
                    landing_bonus + failure_penalty)

    components = {
        'progress': progress,
        'survival': survival,
        'stability': stability,
        'fuel': fuel,
        'landing_bonus': landing_bonus,
        'failure_penalty': failure_penalty
    }

    return float(total_reward), components