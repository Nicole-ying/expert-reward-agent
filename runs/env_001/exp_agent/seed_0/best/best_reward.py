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

    # ── 速率标量 ──
    speed = (vx**2 + vy**2) ** 0.5

    # ═══════════════════════════════════════════
    # 主学习信号：进度增量奖励
    # ═══════════════════════════════════════════
    # 每一步向原点靠近就奖励，远离就惩罚
    progress_delta = dist - next_dist
    # mild clip 防目标附近震荡（skeleton 文档推荐）
    progress_delta = max(-0.5, min(0.5, progress_delta))

    # ═══════════════════════════════════════════
    # 软着陆 proxy：连续 3 因子乘积（无 contact）
    # ═══════════════════════════════════════════
    # 去掉 contact_factor —— 飞行中接触永远为 0，会让整个乘积归零
    # 去掉 ang_vel_factor —— 减少因子数，提升 nonzero_rate
    # 三个因子：靠近原点 + 低速 + 姿态正
    prox_factor = max(0.0, 1.0 - next_dist / 1.0)       # 距离 < 1.0 时有梯度
    vel_factor  = max(0.0, 1.0 - speed / 0.5)            # 速率 < 0.5 时有梯度
    ang_factor  = max(0.0, 1.0 - abs(angle) / 0.3)       # 倾角 < 0.3 时有梯度

    soft_landing_proxy = prox_factor * vel_factor * ang_factor

    # ═══════════════════════════════════════════
    # 轻量稳定性惩罚（大幅削弱，目标 < progress 的 10%）
    # ═══════════════════════════════════════════
    # 只保留角速度惩罚，抑制疯狂旋转
    stability_penalty = -0.005 * abs(ang_vel)

    # ═══════════════════════════════════════════
    # 总奖励
    # ═══════════════════════════════════════════
    w_progress = 5.0
    w_soft     = 1.0
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