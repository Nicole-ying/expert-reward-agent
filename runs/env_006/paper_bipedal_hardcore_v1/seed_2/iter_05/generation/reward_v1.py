def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 基本观测提取 ----------
    hull_angle = obs[0]           # 身体倾角 (rad)
    horizontal_speed = obs[2]     # 质心水平速度

    # 只奖励正向移动 (假设正速度朝前)
    forward_vel = max(0.0, horizontal_speed)

    # ---------- 1. 主学习信号：gate‑controlled forward progress ----------
    beta = 2.0  # 姿态敏感度，beta 越大 gate 越早衰减
    gate = 1.0 / (1.0 + beta * abs(hull_angle))
    forward_reward = 1.0 * forward_vel * gate

    # ---------- 2. 稳定约束：hinge fall‑prevention penalty ----------
    # 只在身体倾角超过安全阈值 (0.6 rad ≈ 34°) 时才施加二次惩罚
    safe_angle = 0.6
    angle_excess = max(0.0, abs(hull_angle) - safe_angle)
    hinge_penalty = -0.5 * (angle_excess ** 2)

    # ---------- 3. 效率约束：轻量力矩代价 ----------
    action_cost = sum(action[i] ** 2 for i in range(4))
    energy_penalty = 0.01 * action_cost

    # ---------- 汇总 ----------
    total_reward = forward_reward + hinge_penalty - energy_penalty

    components = {
        "forward_reward": forward_reward,   # gate 调制后的前进奖励
        "hinge_penalty": hinge_penalty,     # 超出安全倾角时的二次惩罚 (≤0)
        "energy_penalty": -energy_penalty   # 力矩代价 (记录为负值)
    }

    return float(total_reward), components