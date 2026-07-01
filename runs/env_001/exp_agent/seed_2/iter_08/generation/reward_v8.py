def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 诊断与修改说明 ==========
    # 上一轮得分 -37.74，episode_length=1000，0% crash。巨大进步！
    # 
    # 问题：agent 存活满长度但得分差，说明在徘徊不落地。
    # proximity_reward 占 91%（ratio=3.68），shaping_reward 仅 0.051 被淹没。
    # soft_landing_bonus 触发率 18.7% 有进步但不够。
    #
    # 修改1（层次1）：proximity_reward 系数从 10→5，削弱主导信号
    # 修改2（层次1）：shaping_reward 系数从 5→10，增强势能引导
    # 修改3（层次2）：soft_landing_bonus 从连续乘积改为加权和
    #   - 乘积形式中 contact_factor 是瓶颈（两个传感器相乘，一个为0则全0）
    #   - 改为加权和：每个因子独立贡献梯度，不再互相阻塞
    #   - 系数从 3→8，让着陆信号更强
    # 修改4（层次1）：stability_penalty 系数从 0.02/0.01/0.01→0.05/0.03/0.03
    #   - 在靠近目标时施加更多约束，防止徘徊
    
    # ========== 公共计算 ==========
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    current_speed = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    next_speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    next_angle = abs(next_obs[4])
    next_angular_vel = abs(next_obs[5])
    
    # ========== 主信号1：bounded_proximity_reward ==========
    # 1/(1+k*dist)，自动 bounded 在 [0,1]
    # 系数从 10→5，削弱主导信号，给其他信号空间
    k_proximity = 5.0
    proximity_reward = 1.0 / (1.0 + k_proximity * next_dist)
    proximity_reward_scaled = 5.0 * proximity_reward  # 之前是 10.0
    
    # ========== 主信号2：potential_based_shaping ==========
    # Φ = -distance - 0.3*speed - 0.2*abs(angle)
    # F = γ*Φ(next) - Φ(current)
    # 系数从 5→10，增强势能引导
    gamma = 0.99
    phi_current = -(current_dist + 0.3 * current_speed + 0.2 * abs(obs[4]))
    phi_next = -(next_dist + 0.3 * next_speed + 0.2 * next_angle)
    shaping_reward = gamma * phi_next - phi_current
    shaping_reward_scaled = 10.0 * shaping_reward  # 之前是 5.0
    
    # ========== 稳定约束：bounded_stability_penalty ==========
    # 距离门控：只在靠近目标时施加
    gate_radius = 2.0
    distance_gate = max(0.0, 1.0 - next_dist / gate_radius)
    
    # bounded 形式
    speed_penalty_bounded = next_speed / (next_speed + 1.0)
    angle_penalty_bounded = next_angle / (next_angle + 0.5)
    angular_vel_bounded = next_angular_vel / (next_angular_vel + 1.0)
    
    # 系数从 0.02/0.01/0.01 → 0.05/0.03/0.03
    stability_penalty = -distance_gate * (
        0.05 * speed_penalty_bounded + 
        0.03 * angle_penalty_bounded + 
        0.03 * angular_vel_bounded
    )
    
    # ========== 连续 soft_landing_bonus（改为加权和） ==========
    # 之前是乘积形式：contact_factor 为瓶颈，一个为0则全0
    # 改为加权和：每个因子独立贡献梯度
    near_factor = max(0.0, 1.0 - next_dist / 0.5)  # dist<0.5 时>0
    low_speed_factor = max(0.0, 1.0 - next_speed / 0.3)  # speed<0.3 时>0
    stable_angle_factor = max(0.0, 1.0 - next_angle / 0.2)  # angle<0.2 时>0
    contact_factor = next_obs[6] * next_obs[7]  # 两个支撑都接触时接近1
    
    # 加权和：每个因子独立贡献，不再互相阻塞
    # 权重：near=0.3, speed=0.3, angle=0.2, contact=0.2
    soft_landing_score = (
        0.3 * near_factor + 
        0.3 * low_speed_factor + 
        0.2 * stable_angle_factor + 
        0.2 * contact_factor
    )
    # 系数从 3→8，让着陆信号更强
    soft_landing_bonus = 8.0 * soft_landing_score
    
    # ========== 动作代价：energy_penalty ==========
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