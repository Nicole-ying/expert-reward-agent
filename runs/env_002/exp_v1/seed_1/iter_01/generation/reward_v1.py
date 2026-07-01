def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：progress_delta_reward ==========
    # 势能函数：Phi = -|hull_angle| - 0.5 * |horizontal_velocity - target_velocity|
    # 目标水平速度设为 1.0 m/s（合理行走速度）
    target_velocity = 1.0
    
    # 当前势能
    phi_obs = -abs(obs[0]) - 0.5 * abs(obs[2] - target_velocity)
    # 下一时刻势能
    phi_next = -abs(next_obs[0]) - 0.5 * abs(next_obs[2] - target_velocity)
    
    gamma = 0.99  # 折扣因子
    progress_delta = gamma * phi_next - phi_obs
    
    # 缩放系数，使奖励在合理范围
    progress_delta_reward = 2.0 * progress_delta
    
    # ========== 稳定约束：stability_penalty ==========
    # 惩罚：主体角速度过大、垂直速度过大、主体角度过大
    angular_velocity_penalty = -0.5 * abs(next_obs[1])  # 主体角速度
    vertical_velocity_penalty = -0.3 * abs(next_obs[3])  # 垂直速度
    angle_penalty = -1.0 * abs(next_obs[0])  # 主体角度
    
    stability_penalty = angular_velocity_penalty + vertical_velocity_penalty + angle_penalty
    
    # ========== 动作代价：energy_penalty（小权重） ==========
    # 使用动作的平方和作为能量消耗的代理
    energy_penalty = -0.01 * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)
    
    # ========== 总奖励 ==========
    total_reward = progress_delta_reward + stability_penalty + energy_penalty
    
    # ========== 组件字典 ==========
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components