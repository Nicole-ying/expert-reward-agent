def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 1. forward_progress: 主学习信号 - 鼓励向前行走
    #    signal: horizontal_velocity (obs[2])
    #    operator: dense_state_signal (线性正奖励)
    #    使用线性形式而非凸化，因为v1阶段需要鼓励agent探索不同速度
    # ============================================================
    forward_velocity = obs[2]
    forward_reward = 1.0 * forward_velocity  # 系数1.0，使速度值直接作为奖励
    
    # ============================================================
    # 2. balance_maintenance: 健康约束 - 保持身体直立
    #    signals: hull_angle (obs[0]), hull_angular_velocity (obs[1])
    #    operator: dense_state_signal (hinge penalty)
    #    使用hinge形式：只在角度超过安全阈值时惩罚，避免过度约束
    #    安全阈值设为0.5弧度(~28度)，超过后线性增加惩罚
    # ============================================================
    hull_angle = obs[0]
    hull_angular_vel = obs[1]
    
    # 角度惩罚：只在超过0.5弧度时生效
    angle_threshold = 0.5
    angle_penalty = -2.0 * max(0.0, abs(hull_angle) - angle_threshold)
    
    # 角速度阻尼：轻微惩罚角速度，防止快速旋转
    angular_vel_penalty = -0.1 * hull_angular_vel ** 2
    
    balance_penalty = angle_penalty + angular_vel_penalty
    
    # ============================================================
    # 3. soft_health_gate: 将平衡状态作为门控乘到前进奖励上
    #    当身体角度接近危险时，前进奖励逐渐衰减
    #    使用倒数门形式：1 / (1 + k * abs(angle))
    #    这样在角度较小时门控接近1，角度增大时平滑衰减
    # ============================================================
    gate_k = 3.0  # 控制衰减速度
    health_gate = 1.0 / (1.0 + gate_k * abs(hull_angle))
    
    # 应用门控：前进奖励乘以健康门控
    gated_forward_reward = forward_reward * health_gate
    
    # ============================================================
    # 4. 组合最终奖励
    #    total_reward = 门控前进奖励 + 平衡惩罚
    #    门控前进奖励是主要驱动力，平衡惩罚是辅助约束
    # ============================================================
    total_reward = gated_forward_reward + balance_penalty
    
    # ============================================================
    # 组件记录
    # ============================================================
    components = {
        'forward_reward': forward_reward,          # 原始前进奖励（未门控）
        'health_gate': health_gate,                # 健康门控值
        'gated_forward_reward': gated_forward_reward,  # 门控后的前进奖励
        'angle_penalty': angle_penalty,            # 角度越界惩罚
        'angular_vel_penalty': angular_vel_penalty, # 角速度阻尼
        'balance_penalty': balance_penalty         # 总平衡惩罚
    }
    
    return float(total_reward), components