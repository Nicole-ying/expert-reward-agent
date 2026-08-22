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

    # ---- 超参数 ----
    PROGRESS_WEIGHT = 20.0                # 距离缩减奖励的主权重
    SURVIVAL_PENALTY = -0.08              # 每步生存惩罚，激励快速完成
    SUCCESS_BONUS = 200.0                 # 成功着陆的终端奖励
    FAIL_PENALTY = -30.0                  # 坠毁/出界的终端惩罚

    ANGLE_PENALTY = 0.3                   # 身体倾角二次惩罚系数
    ANG_VEL_PENALTY = 0.03                # 角速度二次惩罚系数

    ACTION_FUEL_PENALTY = -0.01           # 引擎动作的燃料惩罚 (action != 0)

    # 推断成功的阈值
    LAND_DIST_THRESHOLD = 0.2             # 距离垫原点欧氏距离 < 0.2
    LAND_SPEED_THRESHOLD = 0.2            # 合速度 < 0.2
    LAND_ANGLE_THRESHOLD = 0.15           # 身体倾角 < 0.15 rad (~8.6°)
    LAND_CONTACT_REQUIRED = True          # 至少一脚接触垫

    # 推断失败的阈值
    X_BOUNDARY = 1.0                      # 水平飞出边界
    # 坠毁条件：有支撑脚接触 + 地面很近 + (倾角过大 或 高速撞击)
    GROUND_Y_CLOSE = 0.15
    CRASH_ANGLE = 0.8                     # ~45°
    CRASH_IMPACT_VEL = 1.5

    # ---- 1. 进展信号：基于距离的 improvement_delta ----
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    progress = PROGRESS_WEIGHT * (dist_cur - dist_next)   # 期望 >0

    # ---- 2. 每步存在惩罚 ----
    survival = SURVIVAL_PENALTY

    # ---- 3. 姿态/稳定惩罚 ----
    stability = -ANGLE_PENALTY * (body_angle_next ** 2) - ANG_VEL_PENALTY * (ang_vel_next ** 2)

    # ---- 4. 燃料效率惩罚 ----
    fuel = ACTION_FUEL_PENALTY if action != 0 else 0.0

    # ---- 5. 终端事件推断（在 episode 最后一步时生效） ----
    # 环境在 terminated=True 的最后一帧也会调用 reward，因此可以在 next_obs 中判断
    # 成功着陆条件（严格）
    dist_to_pad = dist_next
    speed = (x_vel_next ** 2 + y_vel_next ** 2) ** 0.5
    angle_ok = abs(body_angle_next) < LAND_ANGLE_THRESHOLD
    contact_ok = (left_contact > 0.5) or (right_contact > 0.5)
    success = (dist_to_pad < LAND_DIST_THRESHOLD and
               speed < LAND_SPEED_THRESHOLD and
               angle_ok and
               contact_ok)

    # 失败条件
    out_of_bounds = abs(x_next) > X_BOUNDARY
    crash = False
    if (left_contact > 0.5 or right_contact > 0.5):
        close_to_ground = y_next < GROUND_Y_CLOSE
        excessive_tilt = abs(body_angle_next) > CRASH_ANGLE
        high_impact = abs(y_vel_next) > CRASH_IMPACT_VEL
        if close_to_ground and (excessive_tilt or high_impact):
            crash = True

    failure = out_of_bounds or crash

    # 只在 episode 真正终止时才给予终端奖励/惩罚，避免截断时误判
    # 通过 info 中的 'terminated' 字段判断（若可用），否则保守地只在满足明确条件时给
    # 如果无法获取 terminated 信息，我们采用保守策略：同时满足成功条件就给 bonus，
    # 但 crash/out_of_bounds 在每步都可能触发（但通常在这些事件发生时会立即终止）。
    # 为了安全，我们依赖条件本身；crash 和 out_of_bounds 只会在边界步出现，所以直接给惩罚。
    landing_bonus = SUCCESS_BONUS if success else 0.0
    failure_penalty = FAIL_PENALTY if failure else 0.0

    # 注意：同一帧不可能既成功又失败，因为条件互斥（成功需要小距离，失败是出界或 crash）
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