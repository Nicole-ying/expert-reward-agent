def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 1. forward_progress: 主学习信号
    #    使用 horizontal_velocity (obs[2]) 作为核心驱动力
    #    采用凸化形式 (signal**2) 以打破低水平稳态
    # ============================================================
    horizontal_vel = obs[2]
    # 凸化正奖励：鼓励持续前进，避免满足于低速
    forward_progress_reward = 1.0 * (horizontal_vel ** 2)
    
    # ============================================================
    # 2. balance_maintenance: 健康约束
    #    使用 hull_angle (obs[0]) 和 hull_angular_velocity (obs[1])
    #    采用 hinge 形式：只在角度偏离超过阈值时惩罚
    #    阈值设为 0.3 rad (~17度)，在终止边界前提供软约束
    # ============================================================
    hull_angle = obs[0]
    hull_angular_vel = obs[1]
    
    # 角度偏离惩罚：只在 |angle| > 0.3 时生效
    angle_threshold = 0.3
    angle_penalty = -0.5 * max(0.0, abs(hull_angle) - angle_threshold) ** 2
    
    # 角速度惩罚：只在角速度过大时生效
    angular_vel_threshold = 2.0
    angular_vel_penalty = -0.1 * max(0.0, abs(hull_angular_vel) - angular_vel_threshold) ** 2
    
    balance_penalty = angle_penalty + angular_vel_penalty
    
    # ============================================================
    # 3. soft_health_gate: 将 balance 状态作为 gate 乘到主奖励上
    #    当 hull_angle 接近危险值时，主奖励平滑衰减
    #    这比单纯加惩罚更有效，因为 agent 在恶化前就会失去前进动力
    # ============================================================
    # 门控因子：角度偏离越大，门控越小
    # 在 0.0~0.5 rad 范围内线性衰减
    gate_factor = max(0.0, min(1.0, 1.0 - abs(hull_angle) / 0.5))
    
    # 应用门控到主奖励
    gated_forward_reward = forward_progress_reward * gate_factor
    
    # ============================================================
    # 4. 组合总奖励
    # ============================================================
    total_reward = gated_forward_reward + balance_penalty
    
    # ============================================================
    # 组件记录
    # ============================================================
    components = {
        'forward_progress_reward': forward_progress_reward,
        'gated_forward_reward': gated_forward_reward,
        'balance_angle_penalty': angle_penalty,
        'balance_angular_vel_penalty': angular_vel_penalty,
        'gate_factor': gate_factor
    }
    
    return float(total_reward), components