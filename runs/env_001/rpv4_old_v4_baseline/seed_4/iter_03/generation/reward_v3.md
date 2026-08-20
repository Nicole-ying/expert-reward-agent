```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v3。
    
    v2→v3 修改：landing_contact 从 per-step state reward 改为一次性 transition bonus。
    
    问题诊断（v2）：landing_contact 作为 per-step 状态奖励（2.0/腿/步），agent 双足着地后
    每步持续收割 ~4.0，导致 magnitude_share 高达 54%。checkpoint 趋势中 cp70% 出现 824.5 
    的极端 spike，随后剧烈振荡，是典型的"占据好状态持续获奖"exploit。
    
    v3 方案：
    - landing_contact: 改为 transition-based — 仅在每条腿从无接触→有接触的瞬间给予
      一次性 25.0/腿 奖励。消除持续累积 exploit，同时保留着陆引导信号。
    - 其余组件（goal_proximity, settling_stability, attitude_constraint, fuel_efficiency）不变。
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
    dist_improvement = current_dist - next_dist
    proximity_bonus = 1.0 * (2.718281828 ** (-next_dist / 0.5))
    goal_proximity = 3.0 * dist_improvement + proximity_bonus
    
    # ========== 邻近门控（多个组件共用）==========
    prox_gate = 1.0 / (1.0 + 5.0 * next_dist)
    
    # ========== Component B: settling_stability ==========
    settling_multiplier = 0.2 + 2.5 * prox_gate
    settling_stability = -0.3 * next_speed * settling_multiplier
    
    # ========== Component C: attitude_constraint ==========
    next_angle_err = abs(next_body_angle)
    next_ang_vel_mag = abs(next_ang_vel)
    attitude_multiplier = 0.3 + 2.0 * prox_gate
    attitude_constraint = -attitude_multiplier * (0.3 * next_angle_err + 0.1 * next_ang_vel_mag)
    
    # ========== Component D: landing_contact (v3: transition-based) ==========
    # 仅在接近目标 + 低速时，对「从无接触到有接触」的转移瞬间给予一次性奖励。
    # 消除 v2 中 per-step 持续累积导致的 exploitation（cp70% spike=824.5）。
    near_target = next_dist < 0.5
    low_speed = next_speed < 1.0
    
    landing_contact = 0.0
    if near_target and low_speed:
        # 左腿：从无接触 → 有接触
        if left_contact < 0.5 and next_left_contact > 0.5:
            landing_contact += 25.0
        # 右腿：从无接触 → 有接触
        if right_contact < 0.5 and next_right_contact > 0.5:
            landing_contact += 25.0
    
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
```