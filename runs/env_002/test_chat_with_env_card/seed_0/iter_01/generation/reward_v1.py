def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 1. forward_progress: 主学习信号
    #    使用 horizontal_velocity (obs[2]) 作为核心驱动力
    #    采用线性正奖励，鼓励向前行走
    # ============================================================
    horizontal_vel = obs[2]
    forward_progress_reward = 1.0 * horizontal_vel  # 权重1.0，线性正奖励

    # ============================================================
    # 2. balance_maintenance: 稳定/安全约束
    #    使用 hull_angle (obs[0]) 和 hull_angular_velocity (obs[1])
    #    采用 quadratic_penalty 形式，惩罚身体倾斜和角速度
    # ============================================================
    hull_angle = obs[0]
    hull_angular_vel = obs[1]
    
    # 惩罚身体倾斜：角度偏离竖直方向（0为竖直）
    angle_penalty = -0.5 * (hull_angle ** 2)
    # 惩罚角速度过大
    angular_vel_penalty = -0.1 * (hull_angular_vel ** 2)
    
    balance_penalty = angle_penalty + angular_vel_penalty

    # ============================================================
    # 3. 组合总奖励
    #    v1阶段：主学习信号 + 必要平衡约束
    #    不加入 energy_efficiency（留到后续迭代）
    #    不加入 gait_rhythm（留到后续迭代）
    #    不加入 terrain_adaptation（留到后续迭代）
    # ============================================================
    total_reward = forward_progress_reward + balance_penalty

    # ============================================================
    # 4. 组件记录
    # ============================================================
    components = {
        "forward_progress_reward": forward_progress_reward,
        "balance_penalty": balance_penalty,
    }

    return float(total_reward), components