def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress delta（到目标垫中心的距离减少量）
    x_prev, y_prev = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    d_prev = (x_prev**2 + y_prev**2) ** 0.5
    d_next = (x_next**2 + y_next**2) ** 0.5
    progress_delta = d_prev - d_next

    # 稳定约束：大幅削弱权重，避免压制 progress 信号
    # 原权重 (0.1, 0.5, 0.1) 使 penalty 均值 0.147 = progress 的 9x
    # 削弱 ~25x → 目标 penalty 均值 ~0.006 = progress 的 ~40%
    x_vel, y_vel = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    w_vel = 0.004      # 原 0.1 → 削弱 25x
    w_angle = 0.02     # 原 0.5 → 削弱 25x
    w_angvel = 0.004   # 原 0.1 → 削弱 25x
    stability_penalty = (
        w_vel * (abs(x_vel) + abs(y_vel)) +
        w_angle * abs(body_angle) +
        w_angvel * abs(ang_vel)
    )

    # 任务完成软代理：接近中心 + 低速 + 水平 + 双脚着地
    # 当前触发率极低（0.2%）是因 penalty 压制导致无法靠近目标，先不调整
    near = (d_next < 0.1)
    slow = (abs(x_vel) < 0.1 and abs(y_vel) < 0.1)
    level = (abs(body_angle) < 0.1)
    both_feet = (next_obs[6] > 0.5 and next_obs[7] > 0.5)
    soft_landing_bonus_raw = 1.0 if (near and slow and level and both_feet) else 0.0
    soft_landing_bonus = 0.5 * soft_landing_bonus_raw

    total_reward = progress_delta - stability_penalty + soft_landing_bonus

    components = {
        "progress_delta": progress_delta,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "total_reward": total_reward
    }

    return float(total_reward), components