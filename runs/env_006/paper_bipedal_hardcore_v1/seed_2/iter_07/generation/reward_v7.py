def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 基本观测提取 ----------
    hull_angle = obs[0]                 # 身体倾角 (rad)
    hull_angular_velocity = obs[1]      # 身体角速度 (rad/s)
    horizontal_speed = obs[2]           # 质心水平速度

    # 只奖励正向移动
    forward_vel = max(0.0, horizontal_speed)

    # ---------- 1. 主学习信号：动态 gate-controlled forward progress ----------
    # gate 由角度和角速度共同决定，剧烈摇晃 → gate 变小 → 前进奖励骤减
    beta_angle = 2.0
    beta_angvel = 1.0
    gate = 1.0 / (1.0 + beta_angle * abs(hull_angle) + beta_angvel * abs(hull_angular_velocity))
    forward_reward = 1.0 * forward_vel * gate

    # ---------- 2. 效率约束：轻量力矩代价 ----------
    action_cost = sum(action[i] ** 2 for i in range(4))
    energy_penalty = 0.01 * action_cost

    # ---------- 3. 姿态铰链惩罚：接近摔倒的直接信号 ----------
    # 阈值 0.3 rad（约为终止条件 0.5 rad 的 60%），超出后线性惩罚
    hinge_penalty = 0.5 * max(0.0, abs(hull_angle) - 0.3)

    # ---------- 汇总 ----------
    total_reward = forward_reward - energy_penalty - hinge_penalty

    components = {
        "forward_reward": forward_reward,
        "energy_penalty": -energy_penalty,
        "hinge_penalty": -hinge_penalty
    }

    return float(total_reward), components