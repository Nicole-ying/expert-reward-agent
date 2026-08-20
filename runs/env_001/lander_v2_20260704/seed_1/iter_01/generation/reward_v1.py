def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取下一步状态
    nx, ny = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    omega = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 权重和阈值（v1 固定值，后续迭代可调）
    w_dist = 1.0
    w_vel = 0.1
    w_angle = 0.1
    w_angvel = 0.05
    w_bonus = 20.0

    thresh_dist = 0.3
    thresh_vx = 0.2
    thresh_vy = 0.2
    thresh_angle = 0.1

    # 组件1: 距离成本（推动 agent 靠近目标着陆垫）
    distance_cost = -w_dist * (nx ** 2 + ny ** 2)

    # 组件2: 稳定性成本（抑制高速、大角度和高角速度，鼓励平稳接近）
    stability_cost = -(
        w_vel * (vx ** 2 + vy ** 2)
        + w_angle * abs(angle)
        + w_angvel * abs(omega)
    )

    # 组件3: 软着陆完成近似奖励（仅在多条件齐备时给予一次正向激励）
    dist = (nx ** 2 + ny ** 2) ** 0.5
    soft_landing_condition = (
        dist < thresh_dist
        and abs(vx) < thresh_vx
        and abs(vy) < thresh_vy
        and abs(angle) < thresh_angle
        and left_contact > 0.5
        and right_contact > 0.5
    )
    soft_landing_bonus = w_bonus if soft_landing_condition else 0.0

    # 合成总奖励
    total_reward = distance_cost + stability_cost + soft_landing_bonus

    # 组件字典（只包含直接参与 total_reward 的项）
    components = {
        'distance_cost': distance_cost,
        'stability_cost': stability_cost,
        'soft_landing_bonus': soft_landing_bonus
    }

    return float(total_reward), components