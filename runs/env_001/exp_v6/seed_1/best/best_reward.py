def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- 配置参数 ---
    # 主学习信号权重
    w_progress = 1.0

    # 稳定性惩罚 — 大幅削弱（上一轮 ratio 0.68，压制了 progress）
    w_speed = 0.001    # 原 0.01 → 降 10x
    w_angle = 0.001    # 原 0.01 → 降 10x
    w_angvel = 0.0001  # 原 0.001 → 降 10x

    # 软着陆代理 — 从二值改为连续乘积，提供密集梯度
    w_proxy = 0.2
    dist_threshold = 0.5       # 放宽距离阈值，配合连续形式
    speed_threshold = 0.3      # 放宽速度阈值
    angle_threshold = 0.5      # 放宽角度阈值

    # --- 1. 进度差分奖励 (主学习信号) ---
    d_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    d_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = d_current - d_next

    # --- 2. 稳定性惩罚 (大幅削弱，避免压制 progress) ---
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_abs = abs(next_obs[4])
    angvel_abs = abs(next_obs[5])

    stability_penalty = -(w_speed * speed + w_angle * angle_abs + w_angvel * angvel_abs)

    # --- 3. 软着陆代理 (连续乘积 → 密集梯度，非零率从 0.38% 提升) ---
    # 每个因子用 bounded max(0, 1 - x/D) 形式，取值范围 [0, 1]，自动限制
    near_factor = max(0.0, 1.0 - d_next / dist_threshold)
    speed_factor = max(0.0, 1.0 - speed / speed_threshold)
    angle_factor = max(0.0, 1.0 - angle_abs / angle_threshold)
    # 腿部接触：用平均接触量替代二值判断，提供部分梯度（单腿触地也有信号）
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