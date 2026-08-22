def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v3。
    
    核心改动：从线性 state-cost → 二次 state-cost（梯度随状态变化）
    - goal_proximity: 二次距离惩罚，梯度 ∝ distance（远处强、近处弱）
    - settling_stability: 二次速度惩罚，梯度 ∝ speed（高速强、低速弱）
    - landing_contact: 渐进式奖励，软化门控提高 active_rate
    - attitude_constraint: 保持结构，提高远距离基线权重
    
    Args:
        obs: 当前观测 (8,)
        action: 当前动作 (0-3)
        next_obs: 下一观测 (8,)
        original_reward: 未使用
        info: 未使用
        training_progress: 未使用
    """
    
    # ========== 信号提取（使用 next_obs 评估转移后状态） ==========
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    # ========== 辅助量 ==========
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    
    # proximity_gate: 1.0 在目标处，~0.1 在 distance≈3
    proximity_gate = 1.0 / (1.0 + 3.0 * distance)
    
    # ========== Component A: goal_proximity — 二次距离代价 ==========
    # 梯度 = -0.2 * distance，远处梯度强（d=3→-0.6），近处梯度弱（d=0.3→-0.06）
    # 天然 curriculum：远处强力驱动靠近，近处精细调控
    goal_proximity = -0.1 * distance ** 2
    
    # ========== Component B: settling_stability — 二次速度代价 ==========
    # 梯度 = -0.6 * speed * gate，高速时梯度强，且随接近目标增强
    # gate 范围：远距离约 0.28，近距离约 1.0，提供 ~3.6x 动态范围
    settling_stability = -0.3 * speed ** 2 * (0.2 + 0.8 * proximity_gate)
    # 角速度惩罚（新增独立项）
    settling_stability += -0.08 * ang_vel ** 2 * (0.2 + 0.8 * proximity_gate)
    
    # ========== Component C: attitude_constraint — 姿态代价 ==========
    # 保持线性但将远距离基线从 0.15 提高到 0.3，更早开始姿态学习
    angle_error = abs(body_angle)
    ang_vel_mag = abs(ang_vel)
    attitude_gate = 0.3 + 0.7 * proximity_gate
    attitude_constraint = -0.4 * angle_error * attitude_gate
    attitude_constraint += -0.15 * ang_vel_mag * attitude_gate
    
    # ========== Component D: landing_contact — 渐进式着陆奖励 ==========
    # 软化门控：distance<0.5（原 0.3），speed<1.0（原 0.6）
    # 使用连续缩放因子替代二元门控，提供稠密梯度
    near_pad = distance < 0.5
    slow_enough = speed < 1.0
    
    if near_pad and slow_enough:
        # 连续因子：越近越好、越慢越好
        prox_factor = 1.0 - distance / 0.5      # 1.0@d=0, 0.0@d=0.5
        speed_factor = 1.0 - speed / 1.0        # 1.0@s=0, 0.0@s=1.0
        contact_frac = (left_contact + right_contact) / 2.0  # 0.0~1.0
        
        # base_bonus: 靠近+低速即有奖励，引导 agent 先学会接近
        base_bonus = 1.5 * prox_factor * speed_factor
        # contact_bonus: 接触额外奖励，推动完成最后一步
        contact_bonus = 2.0 * prox_factor * speed_factor * contact_frac
        
        landing_contact = base_bonus + contact_bonus
    else:
        landing_contact = 0.0
    
    # ========== Component E: fuel_efficiency — 燃料代价 ==========
    fuel_efficiency = -0.03 * float(action != 0)
    
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