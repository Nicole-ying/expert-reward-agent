def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ====== 主学习信号：线性前进速度奖励（恢复） ======
    forward_vel = next_obs[2]
    w_forward = 1.0
    forward_reward = w_forward * forward_vel  # 回到 iter1 的线性形式

    # ====== 姿态稳定性惩罚（系数放松） ======
    hull_angle = obs[0]
    hull_angvel = obs[1]
    vertical_vel = obs[3]
    w_angle = 0.3        # 原 0.5 → 0.3，避免过罚小幅摆动
    w_angvel = 0.01
    w_vertical = 0.01
    stability_penalty = -(
        w_angle * hull_angle**2 +
        w_angvel * hull_angvel**2 +
        w_vertical * vertical_vel**2
    )

    # ====== 能量效率惩罚（适度加强） ======
    w_energy = 0.01       # 原 0.001 → 0.01，使能耗有感知力
    energy_penalty = -w_energy * (
        action[0]**2 + action[1]**2 +
        action[2]**2 + action[3]**2
    )

    total_reward = forward_reward + stability_penalty + energy_penalty
    components = {
        'forward_reward': forward_reward,
        'stability_penalty': stability_penalty,
        'energy_penalty': energy_penalty
    }
    return float(total_reward), components