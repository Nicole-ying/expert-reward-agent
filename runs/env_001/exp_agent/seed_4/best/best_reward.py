def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- HYPERPARAMETERS ----------
    w_progress = 1.0        # 主学习信号：每步接近目标给予奖励
    w_vel     = 0.003       # 【降低 3.3x】0.01→0.003，减轻速度惩罚对 progress 的抵消
    w_angle   = 0.003       # 【降低 3.3x】0.01→0.003，同上
    w_angvel  = 0.002       # 【降低 2.5x】0.005→0.002，同上
    w_landing = 0.03        # 保持不变：iter 5 降低后 crash，当前值已验证有效

    # 连续 landing proxy 的阈值（bounded 因子归零点）
    D_max = 2.0             # 距离阈值：超过此距离 proximity 归零
    V_max = 3.0             # 速度阈值：超过此速度 smooth_speed 归零
    A_max = 0.5             # 角度阈值（弧度，~28°）：超过此角度 upright 归零

    # 距离门控：stability penalty 只在接近地面时生效
    D_gate = 3.0            # 距离阈值：超过此距离 stability penalty 被门控衰减
    # --------------------------------

    # --- Progress toward target ---
    dist_obs  = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress_delta = dist_obs - dist_next
    progress_reward = w_progress * progress_delta

    # --- Stability penalty (DISTANCE-GATED) ---
    # 门控因子：距离远 → 0（自由机动），距离近 → 1（精细控制）
    vx, vy = next_obs[2], next_obs[3]
    angle  = next_obs[4]
    angvel = next_obs[5]

    abs_v_sum = abs(vx) + abs(vy)
    abs_angle = abs(angle)
    abs_angvel = abs(angvel)

    raw_penalty = w_vel * abs_v_sum + w_angle * abs_angle + w_angvel * abs_angvel
    gate = max(0.0, 1.0 - dist_next / D_gate)  # ∈ [0, 1]，距离越近门越开

    stability_penalty = -raw_penalty * gate

    # --- Soft landing proxy (CONTINUOUS) ---
    # 三个 bounded 因子，每个 ∈ [0, 1]，乘积提供连续梯度
    proximity    = max(0.0, 1.0 - dist_next / D_max)
    smooth_speed = max(0.0, 1.0 - abs_v_sum / V_max)
    upright      = max(0.0, 1.0 - abs_angle / A_max)

    landing_bonus = w_landing * proximity * smooth_speed * upright

    # --- Total reward ---
    total_reward = progress_reward + stability_penalty + landing_bonus

    components = {
        "total_reward": total_reward,
        "progress_delta_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": landing_bonus
    }

    return float(total_reward), components