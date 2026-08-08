def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========================== 常数参数 ==========================
    w_progress = 1.0       # 接近目标的主学习信号权重
    w_vx       = 0.05      # 水平速度惩罚权重（平方）
    w_vy       = 0.05      # 垂直速度惩罚权重（平方）
    w_angle    = 0.1       # 姿态角惩罚权重（平方）
    w_ang_vel  = 0.01      # 角速度惩罚权重（平方）
    w_action   = 0.01      # 引擎使用惩罚（离散动作）

    # ========================== 观测解析 ==========================
    # obs / next_obs 均为 shape (8,) 的数组
    x_cur  = obs[0]
    y_cur  = obs[1]
    vx_cur = obs[2]
    vy_cur = obs[3]
    # 当前步的姿态等（未使用，但保留）
    # angle_cur = obs[4]
    # ang_vel_cur = obs[5]

    x_next  = next_obs[0]
    y_next  = next_obs[1]
    vx_next = next_obs[2]
    vy_next = next_obs[3]
    angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    # 接触标志：next_obs[6], next_obs[7]，本版未直接使用

    # ========================== 距离计算 ==========================
    dist_before = (x_cur**2 + y_cur**2) ** 0.5
    dist_after  = (x_next**2 + y_next**2) ** 0.5

    # ========================== 组件 A: 距离缩减奖励（主学习信号） ==========================
    # 鼓励每一步使飞行器向目标着陆垫靠近
    progress_reward = w_progress * (dist_before - dist_after)

    # ========================== 组件 B: 稳定/安全约束（合并多个二次惩罚） ==========================
    # B1. 水平速度惩罚 – 鼓励靠近目标时水平静止
    penalty_vx = -w_vx * (vx_next ** 2)

    # B2. 垂直速度惩罚 – 抑制过大垂直速度（无论上升还是下降过快都危险/浪费燃料）
    penalty_vy = -w_vy * (vy_next ** 2)

    # B3. 姿态角惩罚 – 鼓励保持水平姿态
    penalty_angle = -w_angle * (angle_next ** 2)

    # B4. 角速度惩罚 – 防止剧烈旋转
    penalty_ang_vel = -w_ang_vel * (ang_vel_next ** 2)

    stability_penalty = penalty_vx + penalty_vy + penalty_angle + penalty_ang_vel

    # ========================== 组件 C: 引擎使用效率惩罚 ==========================
    action_penalty = 0.0
    if action != 0:   # 动作 0 表示所有引擎关闭
        action_penalty = -w_action

    # ========================== 总奖励 ==========================
    total_reward = progress_reward + stability_penalty + action_penalty

    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'action_penalty': action_penalty
    }

    return float(total_reward), components