def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    奖励函数：势能差塑形 + 速度门控接触转移事件奖励。

    核心变换 (quality-gated transition)：
    将 contact_bonus 从「任何双脚接触转移都发奖」改为「仅低速软着陆的接触转移发奖」，
    消除 agent 通过高速弹跳反复触发接触事件刷分的动机。

    Φ(s) = 1 / (1 + cost)，cost 加权综合距离/速度/姿态。
    """
    # ── 当前状态 ──
    x = float(next_obs[0])
    y = float(next_obs[1])
    vx = float(next_obs[2])
    vy = float(next_obs[3])
    angle = float(next_obs[4])
    angvel = float(next_obs[5])
    left = float(next_obs[6])
    right = float(next_obs[7])

    # ── 上一步状态 ──
    px = float(obs[0])
    py = float(obs[1])
    pvx = float(obs[2])
    pvy = float(obs[3])
    p_angle = float(obs[4])
    p_angvel = float(obs[5])
    p_left = float(obs[6])
    p_right = float(obs[7])

    # ── 当前代价 ──
    dist = (x * x + y * y) ** 0.5
    speed = (vx * vx + vy * vy) ** 0.5
    abs_angle = abs(angle)
    abs_angvel = abs(angvel)

    cost = (
        0.5 * dist +
        0.5 * speed +
        2.0 * abs_angle +
        1.0 * abs_angvel
    )

    # ── 上一步代价 ──
    prev_dist = (px * px + py * py) ** 0.5
    prev_speed = (pvx * pvx + pvy * pvy) ** 0.5
    prev_abs_angle = abs(p_angle)
    prev_abs_angvel = abs(p_angvel)

    prev_cost = (
        0.5 * prev_dist +
        0.5 * prev_speed +
        2.0 * prev_abs_angle +
        1.0 * prev_abs_angvel
    )

    # ── 势能函数：Φ(s) = 1/(1+cost)，有界 ∈ (0, 1] ──
    phi = 1.0 / (1.0 + cost)
    prev_phi = 1.0 / (1.0 + prev_cost)

    # ── 势能差塑形：Φ(next) - Φ(prev) ──
    progress_scale = 10.0
    landing_progress = progress_scale * (phi - prev_phi)

    # ── 接触转移事件 + 速度门控 ──
    # 双脚接触转移：从未接触 → 接触
    prev_both = 1.0 if (p_left > 0.5 and p_right > 0.5) else 0.0
    curr_both = 1.0 if (left > 0.5 and right > 0.5) else 0.0
    contact_event = 1.0 if (curr_both > 0.5 and prev_both < 0.5) else 0.0

    # 速度门控：仅低速软着陆时发奖，高速弹跳不给奖励
    safe_speed = 1.5  # 安全着陆速度阈值
    softness = max(0.0, 1.0 - speed / safe_speed)

    contact_scale = 5.0
    contact_bonus = contact_scale * contact_event * phi * softness

    total_reward = landing_progress + contact_bonus

    components = {
        "landing_progress": landing_progress,
        "contact_bonus": contact_bonus
    }

    return float(total_reward), components