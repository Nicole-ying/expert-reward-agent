def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # === 主学习信号：bounded proximity（不变） ===
    x_next, y_next = next_obs[0], next_obs[1]
    d_next = (x_next ** 2 + y_next ** 2) ** 0.5
    k_proximity = 5.0
    bounded_proximity = 1.0 / (1.0 + k_proximity * d_next)

    # === 稳定约束（不变） ===
    x_vel, y_vel = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    w_vel = 0.004
    w_angle = 0.02
    w_angvel = 0.004
    stability_penalty = (
        w_vel * (abs(x_vel) + abs(y_vel)) +
        w_angle * abs(body_angle) +
        w_angvel * abs(ang_vel)
    )

    # === 软着陆奖励：连续乘积，阈值收紧至 0.1（匹配原二值条件） ===
    # 上一轮阈值 0.2 太宽，nonzero_rate 51% 说明 agent 在目标附近悬浮吃部分信用
    # 收紧到 0.1 后，只有真正接近着陆条件时才有非零信号，保留连续梯度优势
    near_factor = max(0.0, 1.0 - d_next / 0.1)
    speed = abs(x_vel) + abs(y_vel)
    slow_factor = max(0.0, 1.0 - speed / 0.1)
    level_factor = max(0.0, 1.0 - abs(body_angle) / 0.1)
    feet_factor = 0.5 * (next_obs[6] + next_obs[7])

    soft_landing_raw = near_factor * slow_factor * level_factor * feet_factor
    soft_landing_bonus = 0.5 * soft_landing_raw  # 权重不变

    total_reward = bounded_proximity - stability_penalty + soft_landing_bonus

    components = {
        "bounded_proximity": bounded_proximity,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
    }

    return float(total_reward), components