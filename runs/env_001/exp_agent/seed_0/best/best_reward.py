def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 诊断与修改说明：
    # 问题：score=-115，100% early terminal。agent 在 progress_delta 引导下
    # 接近目标，但 crash 而非 soft landing。soft_landing_proxy 的 nonzero_rate=0.72%
    # 仍然太稀疏（乘积形式导致任一因子为零则整体为零）。
    #
    # 修改方案：
    # 1. 用 potential_based_shaping 替代 progress_delta_reward。
    #    势能函数 Φ = -(dist + 0.8*speed + 0.5*|angle|)，同时引导接近、减速、稳定。
    #    理论保证最优策略不变（Ng 1999），天然抗震荡。
    # 2. soft_landing_proxy 改为加权和形式（非乘积），提高 nonzero_rate。
    #    每个因子独立贡献梯度，不会因为某个条件不满足而完全消失。
    # 3. stability_penalty 保留角速度惩罚（距离门控），速度和角度已由 shaping 覆盖。
    # ============================================================
    
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    
    current_speed = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    next_speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    
    current_angle = abs(obs[4])
    next_angle = abs(next_obs[4])
    
    # ============================================================
    # 1. 主学习信号: potential_based_shaping
    #    势能 Φ = -(dist + 0.8*speed + 0.5*|angle|)
    #    shaping = γ * Φ(next) - Φ(obs), γ=0.99
    #    同时引导：接近目标 ↓、减速 ↓、姿态稳定 ↓
    # ============================================================
    gamma = 0.99
    phi_obs = -(current_dist + 0.8 * current_speed + 0.5 * current_angle)
    phi_next = -(next_dist + 0.8 * next_speed + 0.5 * next_angle)
    
    shaping_scale = 2.0
    potential_shaping = shaping_scale * (gamma * phi_next - phi_obs)

    # ============================================================
    # 2. 稳定约束: angular_vel_penalty（距离门控）
    #    速度和角度已由 shaping 覆盖，只保留角速度惩罚
    # ============================================================
    angular_vel = abs(next_obs[5])
    gate_radius = 2.0
    distance_gate = max(0.0, 1.0 - next_dist / gate_radius)
    
    angular_vel_penalty_weight = 0.02
    angular_vel_penalty = -angular_vel_penalty_weight * angular_vel * distance_gate

    # ============================================================
    # 3. 任务完成proxy: soft_landing_proxy（加权和形式）
    #    用加权和替代乘积，每个因子独立贡献梯度
    #    提高 nonzero_rate，让 agent 逐步学会各个条件
    # ============================================================
    # 距离因子：dist < 0.8 时开始贡献
    dist_factor = max(0.0, 1.0 - next_dist / 0.8)
    # 速度因子：speed < 0.5 时开始贡献
    speed_factor = max(0.0, 1.0 - next_speed / 0.5)
    # 姿态角因子：angle < 0.3 时开始贡献
    angle_factor = max(0.0, 1.0 - next_angle / 0.3)
    # 接触因子
    contact_factor = min(1.0, (next_obs[6] + next_obs[7]) / 2.0)
    
    # 加权和（各因子权重不同，dist 最重要）
    landing_bonus_weight = 1.5
    soft_landing_proxy = landing_bonus_weight * (
        0.4 * dist_factor + 
        0.3 * speed_factor + 
        0.2 * angle_factor + 
        0.1 * contact_factor
    )

    # ============================================================
    # 4. 动作代价: energy_penalty（小权重）
    # ============================================================
    engine_use = 1.0 if action != 0 else 0.0
    energy_penalty_weight = 0.01
    energy_penalty = -energy_penalty_weight * engine_use

    # ============================================================
    # 总奖励
    # ============================================================
    total_reward = potential_shaping + angular_vel_penalty + soft_landing_proxy + energy_penalty

    components = {
        "potential_shaping": potential_shaping,
        "angular_vel_penalty": angular_vel_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }

    return float(total_reward), components