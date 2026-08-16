def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v4。

    根本性重构（vs v3）：
    - goal_proximity：从「永久正 state bonus + delta」切换为
      **纯势能差分 potential-based shaping**。
      Φ(s) = -w_lin * dist + w_exp * e^(-dist)
      reward = Φ(next_obs) - Φ(obs)
      总奖励有界（≈ w_lin * initial_dist + w_exp），悬停收益=0，振荡净收益=0。
      彻底消除 v3 中 agent 悬停收租的 exploit。

    - settling_stability：降低状态惩罚系数（-1.0 vs v3 的 -1.2），
      让 agent 不再因靠近目标而被惩罚到退缩。

    - attitude_constraint：降低姿态/角速度惩罚系数，扩宽门控范围
      （dist>=2.5 死区 vs v3 的 2.0），让中距离机动不被惩罚。

    - landing_contact：放宽门控（dist<0.4 vs v3 的 0.25，speed<0.6 vs v3 的 0.4），
      提高双支撑奖励至 30.0，让着陆信号真正可触达且足够诱人。
    """

    # ========== 信号提取 ==========
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    ang_vel = obs[5]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_vel = next_obs[5]
    n_left, n_right = next_obs[6], next_obs[7]

    # L2 距离与速率
    current_dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    current_speed = (vx**2 + vy**2) ** 0.5
    next_speed = (nvx**2 + nvy**2) ** 0.5

    # ================================================================
    # Component A: goal_proximity — 纯势能差分 (potential-based shaping)
    #
    #   Φ(s) = -w_lin * dist + w_exp * e^(-dist)
    #   reward = Φ(next) - Φ(current)
    #
    #   数学性质：
    #   - 总 episode 奖励 = Φ(final) - Φ(initial)，路径无关
    #   - 悬停 (dist 不变)：reward = 0
    #   - 振荡 (来回)：净 reward = 0
    #   - 靠近目标：reward > 0；远离：reward < 0
    #
    #   系数设计：
    #   - w_lin=5.0：线性梯度，远处每步 ~0.5-1.0 的正向信号
    #   - w_exp=5.0：指数势能，在目标附近提供陡峭梯度
    #     dist=0→Φ_exp=5.0, dist=0.5→Φ_exp=3.03, dist=1→Φ_exp=1.84
    # ================================================================
    phi_current = -5.0 * current_dist + 5.0 * (2.718281828 ** (-current_dist))
    phi_next    = -5.0 * next_dist    + 5.0 * (2.718281828 ** (-next_dist))
    goal_proximity = phi_next - phi_current

    # ================================================================
    # Component B: settling_stability — 门控速度管理
    #
    #   比 v3 更轻的状态惩罚（-1.0 vs -1.2），降低靠近目标时的阻力。
    #   prox_gate: dist=0→1.0, dist=1→0.33, dist=2→0.20
    # ================================================================
    prox_gate = 1.0 / (1.0 + 2.0 * next_dist)
    speed_improvement = current_speed - next_speed

    # delta：奖励减速，近处权重放大
    speed_delta = 2.5 * speed_improvement * (0.5 + 2.0 * prox_gate)

    # 状态惩罚：轻量线性速度惩罚，仅在接近目标时生效
    speed_state = -1.0 * next_speed * prox_gate

    settling_stability = speed_delta + speed_state

    # ================================================================
    # Component C: attitude_constraint — 扩宽死区的姿态管理
    #
    #   dist >= 2.5：完全不干预（v3 是 2.0），给中距离机动更多自由
    #   dist <= 0.5：全力约束
    #   中间线性过渡
    #   降低惩罚系数以减少靠近目标时的排斥力
    # ================================================================
    if next_dist >= 2.5:
        att_gate = 0.0
    elif next_dist <= 0.5:
        att_gate = 1.0
    else:
        att_gate = (2.5 - next_dist) / 2.0  # 从 1.0 线性降到 0.0

    angle_improvement = abs(angle) - abs(n_angle)
    angvel_improvement = abs(ang_vel) - abs(n_ang_vel)

    angle_delta = att_gate * (2.0 * angle_improvement + 1.0 * angvel_improvement)
    # 降低状态惩罚系数：angle: -2.0(v4) vs -3.0(v3), angvel: -1.0(v4) vs -1.5(v3)
    angle_penalty = att_gate * (-2.0 * abs(n_angle) - 1.0 * abs(n_ang_vel))

    attitude_constraint = angle_delta + angle_penalty

    # ================================================================
    # Component D: landing_contact — 核心正向激励（放宽门控）
    #
    #   放宽条件让着陆信号可触达：
    #   - dist < 0.4（v3 是 0.25）
    #   - speed < 0.6（v3 是 0.4）
    #   提高双支撑奖励至 30.0（v3 是 25.0）
    #   单边接触给 6.0 作为渐进引导
    # ================================================================
    near_target = next_dist < 0.4
    low_speed = next_speed < 0.6
    both_contacts = (n_left > 0.5 and n_right > 0.5)
    any_contact = (n_left > 0.5 or n_right > 0.5)

    landing_contact = 0.0
    if near_target and low_speed:
        if both_contacts:
            landing_contact = 30.0
        elif any_contact:
            landing_contact = 6.0

    # ================================================================
    # Component E: fuel_efficiency — 轻量引擎使用惩罚
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