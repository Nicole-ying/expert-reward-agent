def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v3。

    根本性重构（vs v2）：
    - goal_proximity：从「delta + 线性状态惩罚」切换为「有界指数 proximity bonus + delta」。
      指数衰减 e^(-dist) 在 target 处=1.0，远处→0，始终为正且有界，
      提供了可优化的连续梯度，同时不会被量级淹没。
    - settling_stability：大幅降低状态速度惩罚系数（-1.2 vs v2 的 -4.0），
      提升 delta 权重，使远处基本为零、近处温和引导减速。
    - attitude_constraint：门控阈值放宽到 2.0，让 agent 在中距离就开始关注姿态。
    - landing_contact：收紧条件、提高奖励到 25.0，着陆信号更显著。
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
    # Component A: goal_proximity
    #   有界指数 proximity bonus（核心创新）+
    #   delta 距离改进（即时方向反馈）
    #
    #   proximity_bonus: 3.0 * e^(-dist)
    #     dist=0.0 → 3.00   dist=0.5 → 1.82   dist=1.0 → 1.10
    #     dist=2.0 → 0.41   dist=5.0 → 0.02
    #   提供了从远处到近处的连续可优化梯度，始终为正且有界。
    # ================================================================
    proximity_bonus = 3.0 * (2.718281828 ** (-next_dist))
    dist_improvement = current_dist - next_dist
    goal_proximity = proximity_bonus + 5.0 * dist_improvement

    # ================================================================
    # Component B: settling_stability
    #   接近度门控的速度管理：远处几乎不干预，近处温和引导减速。
    #   prox_gate: dist=0→1.0, dist=1→0.33, dist=2→0.20, dist=5→0.09
    # ================================================================
    prox_gate = 1.0 / (1.0 + 2.0 * next_dist)
    speed_improvement = current_speed - next_speed

    # delta：奖励减速，近处权重放大
    speed_delta = 3.0 * speed_improvement * (0.5 + 2.0 * prox_gate)

    # 状态惩罚：温和的线性速度惩罚，近处显著、远处可忽略
    speed_state = -1.2 * next_speed * prox_gate

    settling_stability = speed_delta + speed_state

    # ================================================================
    # Component C: attitude_constraint
    #   两级门控：dist>=2.0 完全不干预，dist<=0.5 全力约束，中间线性过渡。
    #   比 v2 的 1.5 阈值更宽松，让 agent 在中距离开始关注姿态。
    # ================================================================
    if next_dist >= 2.0:
        att_gate = 0.0
    elif next_dist <= 0.5:
        att_gate = 1.0
    else:
        att_gate = (2.0 - next_dist) / 1.5  # 从 1.0 线性降到 0.0

    angle_improvement = abs(angle) - abs(n_angle)
    angvel_improvement = abs(ang_vel) - abs(n_ang_vel)

    angle_delta = att_gate * (2.5 * angle_improvement + 1.0 * angvel_improvement)
    angle_penalty = att_gate * (-3.0 * abs(n_angle) - 1.5 * abs(n_ang_vel))

    attitude_constraint = angle_delta + angle_penalty

    # ================================================================
    # Component D: landing_contact
    #   收紧条件（dist<0.25, speed<0.4），提高奖励到 25.0。
    #   双支撑着陆每步大奖励，单边接触小鼓励。
    # ================================================================
    near_target = next_dist < 0.25
    low_speed = next_speed < 0.4
    both_contacts = (n_left > 0.5 and n_right > 0.5)
    any_contact = (n_left > 0.5 or n_right > 0.5)

    landing_contact = 0.0
    if near_target and low_speed:
        if both_contacts:
            landing_contact = 25.0
        elif any_contact:
            landing_contact = 5.0

    # ================================================================
    # Component E: fuel_efficiency
    #   轻量引擎使用惩罚，不阻碍必要的探索。
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