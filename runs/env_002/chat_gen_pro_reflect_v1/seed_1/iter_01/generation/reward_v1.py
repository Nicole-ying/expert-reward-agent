def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 信号提取 ==========
    # 主学习信号：水平速度
    horizontal_velocity = obs[2]
    
    # 平衡信号：身体角度和角速度
    hull_angle = obs[0]
    hull_angular_velocity = obs[1]
    
    # 能耗信号：动作扭矩
    hip1_torque = action[0]
    knee1_torque = action[1]
    hip2_torque = action[2]
    knee2_torque = action[3]
    
    # ========== 组件 1: 前进奖励 (主学习信号) ==========
    # 使用 dense_state_signal 的线性正奖励形式
    # 直接奖励水平速度，鼓励 agent 向前移动
    forward_reward_weight = 2.0
    forward_reward = forward_reward_weight * horizontal_velocity
    
    # ========== 组件 2: 平衡约束 (soft_health_gate) ==========
    # 使用 soft_health_gate 将平衡状态作为门控乘到前进奖励上
    # 当身体角度偏离竖直方向或角速度过大时，门控因子衰减
    # 这样 agent 在平衡恶化时不会获得前进奖励，而不是额外惩罚
    angle_threshold = 0.5  # 弧度，约28度
    angular_velocity_threshold = 2.0  # 弧度/秒
    
    # 计算角度偏离惩罚因子 (0~1之间)
    angle_factor = 1.0 / (1.0 + 2.0 * abs(hull_angle))
    # 计算角速度惩罚因子 (0~1之间)
    angular_velocity_factor = 1.0 / (1.0 + 0.5 * abs(hull_angular_velocity))
    
    # 组合门控因子：取最小值确保最差情况主导
    balance_gate = min(angle_factor, angular_velocity_factor)
    
    # 应用门控：前进奖励乘以平衡门控
    gated_forward_reward = forward_reward * balance_gate
    
    # ========== 组件 3: 能耗惩罚 (quadratic_penalty) ==========
    # 使用 quadratic_penalty 惩罚动作扭矩平方和
    # 权重较小，避免压制前进动力
    energy_penalty_weight = 0.01
    energy_penalty = -energy_penalty_weight * (
        hip1_torque**2 + knee1_torque**2 + hip2_torque**2 + knee2_torque**2
    )
    
    # ========== 组件 4: 额外平衡惩罚 (dense_state_signal hinge形式) ==========
    # 当身体角度超过安全阈值时，给予额外惩罚
    # 使用 hinge 形式，只在越界时生效
    angle_safety_threshold = 0.8  # 弧度，约46度
    angle_penalty_weight = 1.0
    angle_penalty = -angle_penalty_weight * max(0.0, abs(hull_angle) - angle_safety_threshold)
    
    # ========== 总奖励 ==========
    total_reward = gated_forward_reward + energy_penalty + angle_penalty
    
    # ========== 组件记录 ==========
    components = {
        "gated_forward_reward": gated_forward_reward,
        "energy_penalty": energy_penalty,
        "angle_penalty": angle_penalty,
    }
    
    return float(total_reward), components