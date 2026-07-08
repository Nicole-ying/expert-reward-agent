def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前观测与下一观测
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 主学习信号：progress delta ----------
    def dist(px, py):
        return (px**2 + py**2)**0.5

    prev_dist = dist(x, y)
    next_dist = dist(next_x, next_y)
    progress_delta = prev_dist - next_dist    # 正值 = 靠近目标

    # ---------- 稳定/安全约束：stability penalty ----------
    speed_penalty = abs(vx) + abs(vy)
    angle_penalty = abs(angle)
    angvel_penalty = abs(angular_vel)
    stability_penalty = -0.1 * speed_penalty - 0.1 * angle_penalty - 0.05 * angvel_penalty

    # ---------- 稠密着陆引导：continuous soft landing proxy ----------
    # 将原来的二元 if 条件替换为连续 bounded 因子
    # 每个因子 ∈ [0, 1]，提供渐进式反馈

    # 距离因子：越靠近目标越好
    proximity_factor = max(0.0, 1.0 - next_dist / 1.0)

    # 速度因子：越慢越好（使用欧氏速度）
    speed = (vx**2 + vy**2)**0.5
    speed_factor = max(0.0, 1.0 - speed / 0.5)

    # 姿态因子：越正越好
    upright_factor = max(0.0, 1.0 - abs(angle) / 0.5)

    # 接触因子：双腿着地更好
    contact_factor = (left_contact + right_contact) / 2.0

    # 综合着陆质量：三个核心因子取几何平均（要求同时满足），
    # 接触因子作为额外加权（避免乘积塌缩到零）
    # 几何平均防止单一因子优秀时掩盖其他因子的不足
    core_quality = (proximity_factor * speed_factor * upright_factor) ** (1.0 / 3.0)
    soft_landing_proxy = 0.5 * core_quality * (0.3 + 0.7 * contact_factor)

    # ---------- 合成 ----------
    total_reward = progress_delta + stability_penalty + soft_landing_proxy

    components = {
        'progress_delta': progress_delta,
        'stability_penalty': stability_penalty,
        'soft_landing_proxy': soft_landing_proxy,
    }
    return float(total_reward), components