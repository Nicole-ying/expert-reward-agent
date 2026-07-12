def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 信号提取 ==========
    horizontal_velocity = obs[2]
    hull_angle = obs[0]
    hull_angular_velocity = obs[1]
    hip1_torque = action[0]
    knee1_torque = action[1]
    hip2_torque = action[2]
    knee2_torque = action[3]

    # ========== 组件 1: 前进奖励 (线性，仅正向) ==========
    # 使用线性速度，避免平方对高速的过度奖励
    forward_reward = 1.0 * max(horizontal_velocity, 0.0)

    # ========== 组件 2: 平衡约束 (soft_health_gate) ==========
    angle_factor = 1.0 / (1.0 + 2.0 * abs(hull_angle))
    angular_velocity_factor = 1.0 / (1.0 + 0.5 * abs(hull_angular_velocity))
    balance_gate = min(angle_factor, angular_velocity_factor)
    gated_forward_reward = forward_reward * balance_gate

    # ========== 组件 3: 能耗惩罚 (quadratic_penalty) ==========
    energy_penalty_weight = 0.01
    energy_penalty = -energy_penalty_weight * (
        hip1_torque**2 + knee1_torque**2 + hip2_torque**2 + knee2_torque**2
    )

    # ========== 组件 4: 额外平衡惩罚 (hinge) ==========
    angle_safety_threshold = 0.8
    angle_penalty_weight = 1.0
    angle_penalty = -angle_penalty_weight * max(0.0, abs(hull_angle) - angle_safety_threshold)

    # ========== 总奖励 ==========
    total_reward = gated_forward_reward + energy_penalty + angle_penalty

    components = {
        "gated_forward_reward": gated_forward_reward,
        "energy_penalty": energy_penalty,
        "angle_penalty": angle_penalty,
    }
    return float(total_reward), components