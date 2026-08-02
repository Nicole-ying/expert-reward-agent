def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 基本观测提取 ----------
    hull_angle = obs[0]                 # 身体倾角 (rad)
    hull_angular_velocity = obs[1]      # 身体角速度 (rad/s)
    horizontal_speed = obs[2]           # 质心水平速度

    # 只奖励正向移动 (假设正速度朝前)
    forward_vel = max(0.0, horizontal_speed)

    # ---------- 1. 主学习信号：动态 gate‑controlled forward progress ----------
    # gate 由角度和角速度共同决定，剧烈摇晃 → gate 变小 → 前进奖励骤减
    beta_angle = 2.0   # 原有角度敏感度
    beta_angvel = 1.0  # 新增角速度敏感度
    gate = 1.0 / (1.0 + beta_angle * abs(hull_angle) + beta_angvel * abs(hull_angular_velocity))
    forward_reward = 1.0 * forward_vel * gate

    # ---------- 2. 效率约束：轻量力矩代价（保持不变） ----------
    action_cost = sum(action[i] ** 2 for i in range(4))
    energy_penalty = 0.01 * action_cost

    # ---------- 汇总（移除原有 hinge_penalty） ----------
    total_reward = forward_reward - energy_penalty

    components = {
        "forward_reward": forward_reward,   # 双因子 gate 调制后的前进奖励
        "energy_penalty": -energy_penalty   # 力矩代价 (记录为负值)
    }

    return float(total_reward), components