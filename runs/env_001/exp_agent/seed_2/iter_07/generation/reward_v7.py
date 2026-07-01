def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 诊断与修改说明 ==========
    # 问题：6轮迭代，最佳得分-108.81，当前-116.95，所有episode提前终止。
    # 核心问题：progress_delta 信号不足以引导agent到达目标区域。
    # 
    # 修改1（换骨架）：用 bounded_proximity_reward 替代 progress_delta_reward
    #   - 数学形态：1/(1+k*dist)，自动 bounded 在 [0,1]
    #   - 靠近目标时信号饱和，不会因距离变化剧烈而震荡
    #   - k=5 使初始距离(~5)处 reward≈0.17，与当前 progress_delta 均值相当
    #
    # 修改2（换骨架）：用 potential_based_shaping 补充梯度
    #   - Φ = -distance - 0.3*speed - 0.2*abs(angle)
    #   - 势能差 F = γ*Φ(next) - Φ(current)，γ=0.99
    #   - 同时引导接近、减速、稳定，且保持最优策略不变
    #
    # 修改3（层次2）：soft_landing_bonus 从二值改为连续乘积
    #   - 每个条件用 max(0, 1-dim/threshold) 形式
    #   - 提供密集梯度，不再依赖<1%的触发率
    #
    # 修改4（层次1）：stability_penalty 进一步削弱并改为 bounded 形式
    #   - 用 1/(1+k*x) 形式使惩罚有上界
    #   - 系数从 0.05/0.03/0.02 降到 0.02/0.01/0.01
    #   - 保持距离门控
    
    # ========== 公共计算 ==========
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    current_speed = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    next_speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    next_angle = abs(next_obs[4])
    next_angular_vel = abs(next_obs[5])
    
    # ========== 主信号1：bounded_proximity_reward ==========
    # 1/(1+k*dist)，自动 bounded 在 [0,1]，k=5
    k_proximity = 5.0
    proximity_reward = 1.0 / (1.0 + k_proximity * next_dist)
    # 放大到与 progress_delta 相当的尺度
    proximity_reward_scaled = 10.0 * proximity_reward
    
    # ========== 主信号2：potential_based_shaping ==========
    # Φ = -distance - 0.3*speed - 0.2*abs(angle)
    # F = γ*Φ(next) - Φ(current)
    gamma = 0.99
    phi_current = -(current_dist + 0.3 * current_speed + 0.2 * abs(obs[4]))
    phi_next = -(next_dist + 0.3 * next_speed + 0.2 * next_angle)
    shaping_reward = gamma * phi_next - phi_current
    # 放大到合理尺度
    shaping_reward_scaled = 5.0 * shaping_reward
    
    # ========== 稳定约束：bounded_stability_penalty ==========
    # 用 1/(1+k*x) 形式使惩罚 bounded，避免数值爆炸
    # 系数大幅降低，保持距离门控
    gate_radius = 2.0
    distance_gate = max(0.0, 1.0 - next_dist / gate_radius)
    
    # bounded 形式的惩罚项，每项在 [0,1) 范围内
    speed_penalty = 1.0 / (1.0 + 1.0 / (next_speed + 0.01))  # 约等于 1 - exp(-speed)
    # 简化：直接用 bounded 线性形式
    speed_penalty_bounded = next_speed / (next_speed + 1.0)  # bounded in [0,1)
    angle_penalty_bounded = next_angle / (next_angle + 0.5)  # bounded in [0,1)
    angular_vel_bounded = next_angular_vel / (next_angular_vel + 1.0)  # bounded in [0,1)
    
    stability_penalty = -distance_gate * (
        0.02 * speed_penalty_bounded + 
        0.01 * angle_penalty_bounded + 
        0.01 * angular_vel_bounded
    )
    
    # ========== 连续 soft_landing_proxy ==========
    # 用连续乘积替代二值 if 条件
    # 每个因子用 max(0, 1-dim/threshold) 形式
    near_factor = max(0.0, 1.0 - next_dist / 0.5)  # dist<0.5 时>0
    low_speed_factor = max(0.0, 1.0 - next_speed / 0.3)  # speed<0.3 时>0
    stable_angle_factor = max(0.0, 1.0 - next_angle / 0.2)  # angle<0.2 时>0
    # contact 用连续值（obs[6], obs[7] 是接触传感器，假设在 [0,1]）
    contact_factor = next_obs[6] * next_obs[7]  # 两个支撑都接触时接近1
    
    # 连续乘积：所有因子相乘，提供密集梯度
    soft_landing_bonus = 3.0 * near_factor * low_speed_factor * stable_angle_factor * contact_factor
    
    # ========== 动作代价：energy_penalty（小权重） ==========
    engine_use = 0.0
    if action == 1 or action == 3:
        engine_use = 0.5
    elif action == 2:
        engine_use = 1.0
    energy_penalty = -0.05 * engine_use
    
    # ========== 总奖励 ==========
    total_reward = proximity_reward_scaled + shaping_reward_scaled + stability_penalty + soft_landing_bonus + energy_penalty
    
    # ========== 组件字典 ==========
    components = {
        "proximity_reward": proximity_reward_scaled,
        "shaping_reward": shaping_reward_scaled,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components