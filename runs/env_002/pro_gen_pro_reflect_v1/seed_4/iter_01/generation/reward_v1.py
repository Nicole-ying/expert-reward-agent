def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -------------------- 核心观测提取 --------------------
    hull_angle     = float(obs[0])   # 身体倾角 (rad)
    hull_ang_vel   = float(obs[1])   # 身体角速度 (rad/s)
    horiz_vel      = float(obs[2])   # 前向速度 (m/s)
    vert_vel       = float(obs[3])   # 竖直速度 (m/s)

    # -------------------- 主进展奖励 (核心驱动力) --------------------
    # 只奖励正的前进速度，负值不惩罚也不奖励
    forward_reward = 1.0 * max(0.0, horiz_vel)

    # -------------------- 姿态与稳定性约束 --------------------
    # 倾角 hinge 惩罚：只在 |angle| > 0.5 rad 时生效，避免安全范围内全时惩罚
    angle_threshold = 0.5
    angle_penalty = -0.5 * max(0.0, abs(hull_angle) - angle_threshold)

    # 角速度二次惩罚：始终有轻量梯度，抑制剧烈摇摆
    angvel_penalty = -0.001 * (hull_ang_vel ** 2)

    # -------------------- 竖直速度抑制（防跳跃） --------------------
    # 步行中正常竖直速度波动 ≤0.5 m/s，超过部分轻罚，避免 agent 通过跳跃刷分
    vert_threshold = 0.5
    vert_penalty = -0.1 * max(0.0, abs(vert_vel) - vert_threshold)

    # -------------------- 总奖励 --------------------
    total_reward = forward_reward + angle_penalty + angvel_penalty + vert_penalty

    components = {
        "forward_reward": forward_reward,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "vert_penalty": vert_penalty,
    }

    return float(total_reward), components