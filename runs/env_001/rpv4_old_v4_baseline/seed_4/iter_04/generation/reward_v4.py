def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v4。

    v3→v4 修改（L1 系数再平衡）：
    
    问题诊断（v3）：landing_contact 从 per-step 改为 transition-based (25.0/腿) 后，
    ep_sum 从 285.6 暴涨至 3090（10.8x），magnitude_share 从 54% 升至 80%。
    agent 学会在垫板上弹跳（contact 反复 0→1→0→1）来收割 transition bonus。
    settling_stability 和 attitude_constraint 被完全淹没（合计 <2%）。
    score 从 178.7 暴跌至 88.5。

    v4 方案：回归 v2 的 per-step landing_contact 机制（方向正确），但大幅降低权重：
    - landing_contact: 2.0/leg → 0.5/leg（降低 4x，消除 dominance）
    - settling_stability: -0.3 → -0.4（提升速度约束可听度）
    - attitude_constraint: 角度权重 0.3 → 0.4（提升姿态约束可听度）
    - goal_proximity 和 fuel_efficiency 不变
    
    预期：landing ~15-20%，goal_proximity ~40-45%，settling ~6-8%，attitude ~3-4%。
    """
    
    # ========== 信号提取 ==========
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact, right_contact = obs[6], obs[7]
    
    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_ang_vel = next_obs[5]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]
    
    # ========== 距离与速度 ==========
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    next_speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    
    # ========== Component A: goal_proximity ==========
    # 改进量（保持 credit assignment 一致性）
    dist_improvement = current_dist - next_dist
    
    # 指数邻近奖励：在目标附近提供每步强梯度
    proximity_bonus = 1.0 * (2.718281828 ** (-next_dist / 0.5))
    
    goal_proximity = 3.0 * dist_improvement + proximity_bonus
    
    # ========== 邻近门控（多个组件共用）==========
    prox_gate = 1.0 / (1.0 + 5.0 * next_dist)
    
    # ========== Component B: settling_stability ==========
    # v4: 速度惩罚系数 -0.3 → -0.4，略微增强约束可听度
    settling_multiplier = 0.2 + 2.5 * prox_gate
    settling_stability = -0.4 * next_speed * settling_multiplier
    
    # ========== Component C: attitude_constraint ==========
    next_angle_err = abs(next_body_angle)
    next_ang_vel_mag = abs(next_ang_vel)
    
    # v4: 角度误差权重 0.3 → 0.4，略微增强姿态约束
    attitude_multiplier = 0.3 + 2.0 * prox_gate
    attitude_constraint = -attitude_multiplier * (0.4 * next_angle_err + 0.1 * next_ang_vel_mag)
    
    # ========== Component D: landing_contact (v4: 回归 per-step，降权) ==========
    # v4: 系数从 2.0/leg 降至 0.5/leg。保持 per-step 机制（v2 验证有效），
    # 但大幅降低量级以消除 dominance 和不稳定振荡。
    near_target = next_dist < 0.5
    low_speed = next_speed < 1.0
    
    landing_contact = 0.0
    if near_target and low_speed:
        landing_contact = 0.5 * (next_left_contact + next_right_contact)
    
    # ========== Component E: fuel_efficiency ==========
    fuel_efficiency = -0.05 * float(action != 0)
    
    # ========== 汇总 ==========
    total_reward = (
        goal_proximity +
        settling_stability +
        attitude_constraint +
        landing_contact +
        fuel_efficiency
    )
    
    components = {
        'goal_proximity': goal_proximity,
        'settling_stability': settling_stability,
        'attitude_constraint': attitude_constraint,
        'landing_contact': landing_contact,
        'fuel_efficiency': fuel_efficiency
    }
    
    return float(total_reward), components