def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v2。
    
    根本性改动（vs v1）：
    - 从纯 improvement-delta 切换为「delta + 直接状态惩罚」混合架构，
      确保每一步都有可优化的梯度信号。
    - 量级整体提升 5~10 倍，使我们的奖励不被环境原生惩罚淹没。
    - 姿态约束门控修复：远距离时完全关闭（gate=0），不再无故惩罚大角度机动。
    - landing_contact 条件放宽 + 奖励提升，使 agent 能感知到着陆信号。
    """
    
    # ========== 信号提取 ==========
    # 当前状态
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    ang_vel = obs[5]
    
    # 下一状态
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_vel = next_obs[5]
    n_left, n_right = next_obs[6], next_obs[7]
    
    # 距离与速率（L2 范数）
    current_dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    current_speed = (vx**2 + vy**2) ** 0.5
    next_speed = (nvx**2 + nvy**2) ** 0.5
    
    # ================================================================
    # Component A: goal_proximity — 混合架构（delta + 直接状态惩罚）
    # ================================================================
    # delta 部分：奖励每一步向目标靠近的进展
    dist_improvement = current_dist - next_dist
    
    # 直接状态惩罚：距离越远惩罚越重，提供逐步梯度
    # 使用平滑函数：dist=0→0，dist=1→−1.5，dist=2→−3.0
    dist_penalty = -1.5 * next_dist
    
    goal_proximity = 8.0 * dist_improvement + dist_penalty
    
    # ================================================================
    # Component B: settling_stability — 速度管理（delta + 直接惩罚）
    # ================================================================
    # 接近度门控：距离越近，速度控制越重要
    # dist=0 → gate≈1.0，dist=1 → gate≈0.33，dist=2 → gate≈0.2
    prox_gate = 1.0 / (1.0 + 2.0 * next_dist)
    
    # delta：奖励减速
    speed_improvement = current_speed - next_speed
    
    # 直接速度惩罚（仅在接近目标时显著）
    speed_penalty = -4.0 * next_speed * prox_gate
    
    settling_stability = 2.0 * speed_improvement * (0.5 + prox_gate) + speed_penalty
    
    # ================================================================
    # Component C: attitude_constraint — 姿态管理（修复门控）
    # ================================================================
    # 两级门控：
    #   dist >= 1.5  → gate=0（完全不干预，允许大角度机动）
    #   dist <= 0.5  → gate=1（全力约束姿态）
    #   中间线性过渡
    if next_dist >= 1.5:
        att_gate = 0.0
    elif next_dist <= 0.5:
        att_gate = 1.0
    else:
        att_gate = 1.0 - (next_dist - 0.5) / 1.0  # 从 1.0 线性降到 0.0
    
    # delta 部分
    angle_improvement = abs(angle) - abs(n_angle)
    angvel_improvement = abs(ang_vel) - abs(n_ang_vel)
    
    # 直接姿态惩罚
    angle_penalty = -att_gate * (abs(n_angle) * 3.0 + abs(n_ang_vel) * 1.5)
    
    # delta 奖励
    angle_delta_reward = att_gate * (2.0 * angle_improvement + 1.0 * angvel_improvement)
    
    attitude_constraint = angle_delta_reward + angle_penalty
    
    # ================================================================
    # Component D: landing_contact — 安全着陆奖励
    # ================================================================
    # 放宽条件，使用连续信号而非硬门控
    near_target = next_dist < 0.3
    low_speed = next_speed < 0.5
    both_contacts = (n_left > 0.5 and n_right > 0.5)
    any_contact = (n_left > 0.5 or n_right > 0.5)
    
    landing_contact = 0.0
    if near_target and low_speed:
        if both_contacts:
            landing_contact = 10.0  # 完美双支撑着陆，每步大奖励
        elif any_contact:
            landing_contact = 2.0   # 单边接触，小鼓励
        # 接近且低速但无接触：不给奖励，不惩罚（让 settling 自然引导）
    
    # ================================================================
    # Component E: fuel_efficiency — 燃料效率
    # ================================================================
    fuel_efficiency = -0.05 * float(action != 0)
    
    # ================================================================
    # 汇总
    # ================================================================
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
        'fuel_efficiency': fuel_efficiency,
    }
    
    return float(total_reward), components