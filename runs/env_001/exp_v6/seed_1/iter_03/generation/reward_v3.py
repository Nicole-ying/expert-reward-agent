def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- 配置参数 ---
    # 主学习信号权重
    w_progress = 1.0

    # 稳定性惩罚 — 保持上一轮的弱水平（ratio 0.088x progress，没问题）
    w_speed = 0.001
    w_angle = 0.001
    w_angvel = 0.0001

    # 软着陆代理 — 连续乘积形式保留，但大幅降权（上一轮 ratio 26.7x progress，压倒主信号）
    w_proxy = 0.008    # 原 0.2 → 降 25x，预期 soft_proxy ~ 0.004，与 progress 持平
    dist_threshold = 0.5
    speed_threshold = 0.3
    angle_threshold = 0.5

    # --- 1. 进度差分奖励 (主学习信号) ---
    d_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    d_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = d_current - d_next

    # --- 2. 稳定性惩罚 ---
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_abs = abs(next_obs[4])
    angvel_abs = abs(next_obs[5])

    stability_penalty = -(w_speed * speed + w_angle * angle_abs + w_angvel * angvel_abs)

    # --- 3. 软着陆代理 (连续乘积，密集梯度) ---
    near_factor = max(0.0, 1.0 - d_next / dist_threshold)
    speed_factor = max(0.0, 1.0 - speed / speed_threshold)
    angle_factor = max(0.0, 1.0 - angle_abs / angle_threshold)
    contact_factor = (next_obs[6] + next_obs[7]) / 2.0

    soft_proxy = w_proxy * near_factor * speed_factor * angle_factor * contact_factor

    # --- 总奖励 ---
    total_reward = progress + stability_penalty + soft_proxy

    components = {
        'progress': progress,
        'stability_penalty': stability_penalty,
        'soft_proxy': soft_proxy,
        'total_reward': total_reward
    }

    return float(total_reward), components