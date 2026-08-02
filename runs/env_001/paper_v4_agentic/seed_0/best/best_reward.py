def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前状态（用于距离计算）
    x, y = obs[0], obs[1]
    # 下一状态
    nx, ny = next_obs[0], next_obs[1]
    nx_v, ny_v = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_v = next_obs[5]
    l_contact = next_obs[6]
    r_contact = next_obs[7]

    # ---------- 1. 进度奖励：向目标 (0,0) 靠近 ----------
    dist_curr = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress = dist_curr - dist_next
    progress_reward = 1.0 * progress

    # ---------- 2. 着陆预备与接触奖励 ----------
    prox = 1.0 / (1.0 + 10.0 * dist_next)
    speed_factor = 1.0 / (1.0 + 5.0 * (abs(nx_v) + abs(ny_v)))
    angle_factor = 1.0 / (1.0 + 3.0 * (abs(n_angle) + abs(n_ang_v)))

    # 双腿接触作为连续因子（二值乘积，0 或 1）
    contact_factor = l_contact * r_contact

    # 混合奖励：未接触时保留一半引导，接触时获得完整奖励
    approach_bonus = 2.0 * prox * speed_factor * angle_factor * (0.5 + 0.5 * contact_factor)

    # ---------- 3. 着陆安全性惩罚 ----------
    v_limit = 0.5
    vx_pen = max(0.0, abs(nx_v) - v_limit)
    vy_pen = max(0.0, abs(ny_v) - v_limit)
    vel_pen = vx_pen + vy_pen

    ang_limit = 1.0
    ang_pen = max(0.0, abs(n_ang_v) - ang_limit)

    tilt_pen = abs(n_angle)

    gate_safety = 1.0 / (1.0 + 5.0 * dist_next)
    landing_safety_penalty = (0.03 * vel_pen + 0.02 * ang_pen + 0.03 * tilt_pen) * gate_safety

    # ---------- 总奖励 ----------
    total_reward = progress_reward + approach_bonus - landing_safety_penalty

    components = {
        "progress_reward": float(progress_reward),
        "approach_bonus": float(approach_bonus),
        "landing_safety_penalty": float(landing_safety_penalty)
    }
    return float(total_reward), components