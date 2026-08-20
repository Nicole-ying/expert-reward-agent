def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── 提取状态 ──
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]

    # ── 距离 ──
    dist = (x**2 + y**2) ** 0.5
    next_dist = (next_x**2 + next_y**2) ** 0.5
    speed = (vx**2 + vy**2) ** 0.5

    # ── 课程参数：training_progress ∈ [0, 1] ──
    t = training_progress
    # 早期容忍度大（鼓励探索），后期收紧（追求精度）
    dist_tolerance = 1.0 + 0.5 * (1.0 - t)   # 1.5 → 1.0
    vel_tolerance  = 0.5 + 0.3 * (1.0 - t)   # 0.8 → 0.5
    ang_tolerance  = 0.3 + 0.2 * (1.0 - t)   # 0.5 → 0.3

    # ═══════════════════════════════════════════
    # 主学习信号：进度增量奖励（保持骨架）
    # ═══════════════════════════════════════════
    progress_delta = dist - next_dist
    progress_delta = max(-0.5, min(0.5, progress_delta))

    # ═══════════════════════════════════════════
    # 软着陆 proxy：连续 3 因子乘积 + 课程收紧
    # ═══════════════════════════════════════════
    prox_factor = max(0.0, 1.0 - next_dist / dist_tolerance)
    vel_factor  = max(0.0, 1.0 - speed / vel_tolerance)
    ang_factor  = max(0.0, 1.0 - abs(angle) / ang_tolerance)

    soft_landing_proxy = prox_factor * vel_factor * ang_factor

    # ═══════════════════════════════════════════
    # 轻量稳定性惩罚
    # ═══════════════════════════════════════════
    stability_penalty = -0.005 * abs(ang_vel)

    # ═══════════════════════════════════════════
    # 总奖励：progress 权重提升，soft 权重削弱
    # ═══════════════════════════════════════════
    # v4 问题：w_soft=1.0 导致 soft 贡献是 progress 的 27x（加权后）
    # v5 修正：w_progress ↑、w_soft ↓，让 progress 成为真正的主信号
    w_progress = 10.0
    w_soft     = 0.15
    w_stab     = 1.0

    total_reward = (
        w_progress * progress_delta +
        w_soft     * soft_landing_proxy +
        w_stab     * stability_penalty
    )

    components = {
        "progress_delta_reward": progress_delta,
        "soft_landing_proxy": soft_landing_proxy,
        "stability_penalty": stability_penalty,
    }

    return float(total_reward), components