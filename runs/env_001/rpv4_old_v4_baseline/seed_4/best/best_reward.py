def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v2。
    核心改进：混合信号框架 — improvement delta + 直接状态评估。
    
    问题诊断（v1）：纯 improvement-delta 导致每 episode 总奖励被初始/终止状态锁死，
    agent 无法通过改进策略影响总奖励，所有组件无学习趋势。
    
    v2 方案：
    - goal_proximity: 保留 delta + 新增指数邻近奖励（创建"漏斗"效应）
    - settling_stability: 改为直接速度惩罚 × 邻近门控
    - attitude_constraint: 改为直接角度惩罚 × 邻近门控
    - landing_contact: 放宽门控条件，提升奖励幅度
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
    # exp(-dist/0.5): dist=0→1.0, dist=0.5→0.37, dist=1.0→0.14, dist=2.0→0.02
    proximity_bonus = 1.0 * (2.718281828 ** (-next_dist / 0.5))
    
    goal_proximity = 3.0 * dist_improvement + proximity_bonus
    
    # ========== 邻近门控（多个组件共用）==========
    # prox_gate: 在目标处≈1.0，远处≈0.1
    prox_gate = 1.0 / (1.0 + 5.0 * next_dist)
    
    # ========== Component B: settling_stability ==========
    # 直接速度惩罚，邻近目标时大幅增强
    # 远处: ~0.2+0.23=0.43 倍，近处: ~0.2+2.5=2.7 倍
    settling_multiplier = 0.2 + 2.5 * prox_gate
    settling_stability = -0.3 * next_speed * settling_multiplier
    
    # ========== Component C: attitude_constraint ==========
    next_angle_err = abs(next_body_angle)
    next_ang_vel_mag = abs(next_ang_vel)
    
    # 直接角度+角速度惩罚，邻近目标时增强
    attitude_multiplier = 0.3 + 2.0 * prox_gate
    attitude_constraint = -attitude_multiplier * (0.3 * next_angle_err + 0.1 * next_ang_vel_mag)
    
    # ========== Component D: landing_contact ==========
    # 放宽门控条件，提高可触发率
    near_target = next_dist < 0.5
    low_speed = next_speed < 1.0
    
    landing_contact = 0.0
    if near_target and low_speed:
        # 双接触给满分，单接触给一半
        landing_contact = 2.0 * (next_left_contact + next_right_contact)
    
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