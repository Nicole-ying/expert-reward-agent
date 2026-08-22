def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # =========================================================================
    # 1. 提取观测信号
    # =========================================================================
    # obs 索引 (共24维)
    hull_angle = obs[0]           # 躯干倾角 (rad)
    hull_ang_vel = obs[1]         # 躯干角速度 (rad/s)
    horizontal_speed = obs[2]     # 水平速度 (m/s)
    vertical_speed = obs[3]       # 垂直速度 (m/s)
    # obs[4..11]: 关节角度和角速度，暂不直接使用
    leg_1_contact = obs[12]       # 腿1触地 (0/1)
    leg_2_contact = obs[13]       # 腿2触地 (0/1)
    # obs[14..23]: lidar, 不可用于奖励

    # next_obs 索引
    next_hull_angle = next_obs[0]
    next_hull_ang_vel = next_obs[1]
    next_horizontal_speed = next_obs[2]
    next_vertical_speed = next_obs[3]
    next_leg_1_contact = next_obs[12]
    next_leg_2_contact = next_obs[13]

    # =========================================================================
    # 2. 前向速度奖励 (主学习信号)
    #    role: forward_velocity_reward
    #    formula: dense_state_signal (线性), 直接鼓励保持正水平速度
    #    scale: 1.0 使每步贡献约 0.8~2.0
    # =========================================================================
    forward_speed = max(0.0, horizontal_speed)  # 负向速度不奖励
    forward_reward = 1.0 * forward_speed

    # =========================================================================
    # 3. 姿态稳定门 (soft_health_gate)
    #    role: upright_penalty 变形为 gate, 替代独立惩罚
    #    rationale: 尝试过的独立 tilt penalty 效果不佳 (score -49~-62).
    #               采用 soft gate 在姿态恶化时直接衰减 forward_reward,
    #               避免 agent 在倾斜时仍因高速获得大量奖励。
    #    formula: soft_health_gate (线性衰减)
    #    gate = 1.0 当 hull_angle 在安全范围内,
    #           线性衰减至 0.0 当 hull_angle 接近危险阈值
    # =========================================================================
    # 设定安全区和衰退区间
    tilt_safe_bound = 0.3          # rad, 近似17°, 正常行走摆动范围
    tilt_danger_bound = 0.7        # rad, 近似40°, 接近摔倒临界 (经验阈值 ~0.8)
    tilt_margin = tilt_danger_bound - tilt_safe_bound  # 0.4 rad 衰退区间

    abs_tilt = abs(hull_angle)
    if abs_tilt <= tilt_safe_bound:
        tilt_gate = 1.0
    elif abs_tilt >= tilt_danger_bound:
        tilt_gate = 0.0
    else:
        tilt_gate = 1.0 - (abs_tilt - tilt_safe_bound) / tilt_margin

    # 角速度惩罚: 当躯干快速旋转时进一步收紧 gate, 捕捉突然失去平衡的前兆
    # 在 tilt_gate 基础上再乘一个角速度衰减因子
    ang_vel_thresh = 2.0           # rad/s, 正常步态摆动通常 < 1.5
    ang_vel_margin = 4.0           # 2.0~6.0 rad/s 区间衰减
    abs_ang_vel = abs(hull_ang_vel)
    if abs_ang_vel <= ang_vel_thresh:
        ang_vel_factor = 1.0
    elif abs_ang_vel >= ang_vel_thresh + ang_vel_margin:
        ang_vel_factor = 0.3       # 不归零, 保留微弱梯度以防止完全丧失学习信号
    else:
        ang_vel_factor = 1.0 - 0.7 * (abs_ang_vel - ang_vel_thresh) / ang_vel_margin

    stability_gate = tilt_gate * ang_vel_factor

    # =========================================================================
    # 4. 能量效率惩罚 (轻量)
    #    role: energy_penalty
    #    formula: action_efficiency (L2 范数)
    #    scale: 极小权重, 仅在主任务已驱动后提供效率偏好
    # =========================================================================
    action_sq_sum = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = 0.005 * action_sq_sum  # 步贡献约 -0.005~-0.02

    # =========================================================================
    # 5. 组合并返回
    # =========================================================================
    # 核心思路: 主奖励通过稳定性 gate 衰减, 再减去轻量能耗
    gated_forward = forward_reward * stability_gate
    total_reward = gated_forward - energy_penalty

    components = {
        "forward_reward": forward_reward,
        "stability_gate": stability_gate,
        "gated_forward": gated_forward,
        "energy_penalty": -energy_penalty
    }

    return float(total_reward), components