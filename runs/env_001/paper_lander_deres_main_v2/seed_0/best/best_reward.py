def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    奖励函数：势能差塑形 + 接触转移事件奖励。

    核心变换 (persistent_to_transition_event)：
    将 contact_bonus 从「每步双脚接触都发奖」改为「仅在双脚从未接触到接触的转移步发奖」，
    消除 agent 在垫上驻留/弹跳刷分的动机。

    Φ(s) = 1 / (1 + cost)，cost 加权综合距离/速度/姿态。
    """
    # ── 当前状态 ──
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left = next_obs[6]
    right = next_obs[7]

    # ── 上一步状态 ──
    px = obs[0]
    py = obs[1]
    pvx = obs[2]
    pvy = obs[3]
    p_angle = obs[4]
    p_angvel = obs[5]
    p_left = obs[6]
    p_right = obs[7]

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

    # ── 接触转移事件奖励：仅在双脚从「未接触」→「接触」的转移步发奖 ──
    prev_both = p_left * p_right      # 上一步双脚接触标志
    curr_both = left * right          # 当前步双脚接触标志
    contact_event = float(curr_both > 0.5 and prev_both < 0.5)  # 转移事件

    contact_scale = 5.0
    contact_bonus = contact_scale * contact_event * phi

    total_reward = landing_progress + contact_bonus

    components = {
        "landing_progress": landing_progress,
        "contact_bonus": contact_bonus
    }

    return float(total_reward), components