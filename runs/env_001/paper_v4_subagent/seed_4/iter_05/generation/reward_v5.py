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

    ANGLE_PENALTY = 0.3
    ANG_VEL_PENALTY = 0.05
    ACTION_PENALTY = 0.05

    SOFT_LANDING_WEIGHT = 0.5
    SOFT_POS_THRESH = 0.3
    SOFT_VEL_THRESH = 0.5
    SOFT_ANGLE_THRESH = 0.2

    SUCCESS_REWARD = 100.0
    SUCCESS_X_THRESH = 0.1
    SUCCESS_Y_THRESH = 0.1
    SUCCESS_VEL_THRESH = 0.2
    SUCCESS_ANGLE_THRESH = 0.1

    # 软安全门控参数
    BOUNDARY_INNER = 0.7      # 开始衰减的内侧距离
    BOUNDARY_LIMIT = 1.0      # 环境视口边界（硬终止）
    CRASH_ANGLE = 0.5
    CRASH_VEL = 1.0
    CRASH_DIST = 0.5
    CRASH_GATE_MIN = 0.2      # 最高危险时 crash_gate 的下限

    # ========== 1. 主进展信号 ==========
    dist_cur = (x_cur**2 + y_cur**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = PROGRESS_WEIGHT * (dist_cur - dist_next)

    # ========== 2. 安全门控（替代原来的硬失败惩罚）==========
    # 边界门：线性衰减，在边界处为 0
    dist_to_boundary = BOUNDARY_LIMIT - abs(x_next)
    if dist_to_boundary <= 0:
        boundary_gate = 0.0
    elif dist_to_boundary < (BOUNDARY_LIMIT - BOUNDARY_INNER):
        boundary_gate = max(0.0, dist_to_boundary / (BOUNDARY_LIMIT - BOUNDARY_INNER))
    else:
        boundary_gate = 1.0

    # 坠毁危险门：仅在有脚接触时评估，连续化
    contact = (left_contact == 1.0) or (right_contact == 1.0)
    if contact:
        angle_danger = min(abs(body_angle_next) / CRASH_ANGLE, 1.0)
        vel_danger = min((abs(x_vel_next) + abs(y_vel_next)) / (2.0 * CRASH_VEL), 1.0)
        dist_danger = min(dist_next / CRASH_DIST, 1.0)
        # 综合危险度（四个指标等权，平均后截断）
        danger = (angle_danger + vel_danger + dist_danger) / 3.0
        danger = min(danger, 1.0)
        crash_gate = 1.0 - (1.0 - CRASH_GATE_MIN) * danger
    else:
        crash_gate = 1.0

    safety_factor = boundary_gate * crash_gate
    # 安全调整量：为负值或零，扮演原来的 failure_penalty 角色
    safety_penalty = progress * (safety_factor - 1.0)

    # ========== 3. 姿态稳定惩罚 ==========
    stability_penalty = (-ANGLE_PENALTY * (body_angle_next ** 2) 
                         - ANG_VEL_PENALTY * (ang_vel_next ** 2))

    # ========== 4. 动作效率惩罚 ==========
    action_penalty = 0.0
    if action != 0:  # 非 no_engine 动作
        action_penalty = -ACTION_PENALTY

    # ========== 5. 软着陆塑造 ==========
    pos_factor = max(0.0, 1.0 - dist_next / SOFT_POS_THRESH)
    vel_abs_sum = abs(x_vel_next) + abs(y_vel_next)
    vel_factor = max(0.0, 1.0 - vel_abs_sum / SOFT_VEL_THRESH)
    angle_factor = max(0.0, 1.0 - abs(body_angle_next) / SOFT_ANGLE_THRESH)
    soft_landing = SOFT_LANDING_WEIGHT * pos_factor * vel_factor * angle_factor

    # ========== 6. 成功着陆奖励 ==========
    landed = (abs(x_next) < SUCCESS_X_THRESH and
              abs(y_next) < SUCCESS_Y_THRESH and
              abs(x_vel_next) < SUCCESS_VEL_THRESH and
              abs(y_vel_next) < SUCCESS_VEL_THRESH and
              abs(body_angle_next) < SUCCESS_ANGLE_THRESH and
              contact)
    success_reward = SUCCESS_REWARD if landed else 0.0

    # ========== 汇总 ==========
    total_reward = (progress + safety_penalty + stability_penalty +
                    action_penalty + soft_landing + success_reward)

    components = {
        'progress': progress,
        'safety_penalty': safety_penalty,      # 替换原 failure_penalty
        'stability_penalty': stability_penalty,
        'action_penalty': action_penalty,
        'soft_landing': soft_landing,
        'success_reward': success_reward
    }

    return float(total_reward), components