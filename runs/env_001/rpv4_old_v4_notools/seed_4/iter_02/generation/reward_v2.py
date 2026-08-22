def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v2 — 完全重构。
    采用势能差 + 全局速度/姿态惩罚 + 高着陆奖励框架，
    引导主体受控下降并安全停靠。
    """
    # 当前状态
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact, right_contact = obs[6], obs[7]

    # 下一状态
    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_ang_vel = next_obs[5]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]

    # ========== A. 目标接近 shaping ==========
    # 势能：对 L1 距离取负，距离越小势能越高
    current_potential = -(abs(x_pos) + abs(y_pos))
    next_potential = -(abs(next_x_pos) + abs(next_y_pos))
    goal_proximity = 10.0 * (next_potential - current_potential)  # 距离减小得正奖励

    # ========== B. 速度惩罚 ==========
    current_speed = abs(x_vel) + abs(y_vel)
    next_speed = abs(next_x_vel) + abs(next_y_vel)
    # 对全局速度进行惩罚，鼓励减速；这里用均值，防止单步过大抖动
    speed_penalty = -0.2 * (current_speed + next_speed) / 2.0

    # ========== C. 姿态约束 ==========
    # 惩罚偏离竖直的绝对角度和角速度
    angle_penalty = -0.5 * abs(next_body_angle)
    ang_vel_penalty = -0.2 * abs(next_ang_vel)
    attitude_constraint = angle_penalty + ang_vel_penalty

    # ========== D. 安全着陆接触奖励 ==========
    near_center = abs(next_x_pos) < 0.2 and abs(next_y_pos) < 0.2
    low_speed_contact = next_speed < 0.5
    both_contacts = (next_left_contact > 0.5 and next_right_contact > 0.5)

    landing_contact = 0.0
    if near_center and low_speed_contact and both_contacts:
        landing_contact = 10.0  # 持续鼓励在目标上双足接触、低速

    # ========== E. 燃料效率 ==========
    # 对任何非悬停动作施以轻惩罚（0: no_engine）
    fuel_efficiency = -0.05 * float(action != 0)

    # ========== 汇总 ==========
    total_reward = (
        goal_proximity +
        speed_penalty +
        attitude_constraint +
        landing_contact +
        fuel_efficiency
    )

    components = {
        'goal_proximity': goal_proximity,
        'speed_penalty': speed_penalty,        # settling_stability 并入速度惩罚
        'attitude_constraint': attitude_constraint,
        'landing_contact': landing_contact,
        'fuel_efficiency': fuel_efficiency
    }

    return float(total_reward), components