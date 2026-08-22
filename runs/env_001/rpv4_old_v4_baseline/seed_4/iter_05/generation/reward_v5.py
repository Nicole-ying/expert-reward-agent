def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v5。

    v4→v5 修改（L1 系数调整）：
    
    问题诊断（v4）：landing_contact 从 v2 的 2.0/腿 被压到 0.5/腿 后，
    max landing = 1.0/步 恰好等于 proximity_bonus = 1.0/步。agent 失去
    「着陆优于盘旋」的信号梯度，选择在目标区低速盘旋而非完成着陆。
    导致 episode 从 v2 的 404 步拉长到 663 步，3/20 截断，score 从 178.7 降至 172.9。
    
    v2 的 54% landing share 被误判为 dominance，但实际上 v2 任务完成最好
    （最快、最可靠）。着陆本身就是任务终点，高 share 反映任务质量而非 exploit。
    
    v5 方案：landing_contact 从 0.5/腿 → 1.0/腿（v2 的一半，v4 的两倍）。
    恢复「着陆 > 盘旋」的梯度（max 2.0 vs 1.0），但不回到 v2 的极端（4.0 vs 1.0）。
    其余组件不变。
    
    预期：landing ~30-35%，goal_proximity ~50-55%，episode 缩短，score → 190+。
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
    settling_multiplier = 0.2 + 2.5 * prox_gate
    settling_stability = -0.4 * next_speed * settling_multiplier
    
    # ========== Component C: attitude_constraint ==========
    next_angle_err = abs(next_body_angle)
    next_ang_vel_mag = abs(next_ang_vel)
    
    attitude_multiplier = 0.3 + 2.0 * prox_gate
    attitude_constraint = -attitude_multiplier * (0.4 * next_angle_err + 0.1 * next_ang_vel_mag)
    
    # ========== Component D: landing_contact (v5: 1.0/腿) ==========
    # v5: 系数从 0.5/腿 → 1.0/腿。恢复「着陆优于盘旋」梯度。
    # 在目标附近 + 低速时，双足着陆 = 2.0/步 > proximity_bonus 1.0/步，
    # 给予 agent 明确动机完成着陆而非盘旋。
    near_target = next_dist < 0.5
    low_speed = next_speed < 1.0
    
    landing_contact = 0.0
    if near_target and low_speed:
        landing_contact = 1.0 * (next_left_contact + next_right_contact)
    
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