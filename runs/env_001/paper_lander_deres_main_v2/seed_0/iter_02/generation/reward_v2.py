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

    # ---------- 距离门控：稳定性惩罚仅在靠近目标时激活 ----------
    speed_penalty = abs(vx) + abs(vy)
    angle_penalty = abs(angle)
    angvel_penalty = abs(angular_vel)

    # 门控因子：dist=0 时为 1.0（全效），dist>=1.5 时为 0.0（关闭）
    gate_threshold = 1.5
    stability_gate = max(0.0, 1.0 - next_dist / gate_threshold)

    stability_penalty = stability_gate * (
        -0.1 * speed_penalty - 0.1 * angle_penalty - 0.05 * angvel_penalty
    )

    # ---------- 任务完成近似信号：soft landing proxy ----------
    both_contact = left_contact + right_contact   # 两个都接触时 = 2.0
    near_target = abs(next_x) < 0.2 and abs(next_y) < 0.2
    low_speed = (abs(vx) + abs(vy)) < 0.15
    upright = abs(angle) < 0.2
    if both_contact == 2.0 and near_target and low_speed and upright:
        soft_landing_proxy = 0.5
    else:
        soft_landing_proxy = 0.0

    # ---------- 合成 ----------
    total_reward = progress_delta + stability_penalty + soft_landing_proxy

    components = {
        'progress_delta': progress_delta,
        'stability_penalty': stability_penalty,
        'soft_landing_proxy': soft_landing_proxy,
    }
    return float(total_reward), components