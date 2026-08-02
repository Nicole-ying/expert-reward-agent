def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 观测拆分
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    angvel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    nangle = next_obs[4]
    nangvel = next_obs[5]
    nleft_contact = next_obs[6]
    nright_contact = next_obs[7]

    # 超参数
    w_progress = 5.0
    w_landing = 2.0
    w_land_vel = 10.0
    w_angle = 0.5
    w_angvel = 0.5
    engine_cost = 0.02

    # 距离计算
    dist = (x**2 + y**2) ** 0.5
    ndist = (nx**2 + ny**2) ** 0.5

    # 1. 主学习信号：距离改进（potential‑based shaping）
    progress = w_progress * (dist - ndist)

    # 2. 着陆质量软信号（仅在双腿接触时激活）
    contact = nleft_contact * nright_contact  # 0 或 1
    x_thresh = 0.1
    vx_thresh = 0.2
    vy_thresh = 0.2
    angle_thresh = 0.1

    fx = max(0.0, 1.0 - abs(nx) / x_thresh)
    fvx = max(0.0, 1.0 - abs(nvx) / vx_thresh)
    fvy = max(0.0, 1.0 - abs(nvy) / vy_thresh)
    fangle = max(0.0, 1.0 - abs(nangle) / angle_thresh)
    fcontact = float(contact)

    if fcontact > 0.5 and fx > 0 and fvx > 0 and fvy > 0 and fangle > 0:
        landing_quality = (fcontact * fx * fvx * fvy * fangle) ** (1.0 / 5.0)
    else:
        landing_quality = 0.0
    landing_reward = w_landing * landing_quality

    # 3. 着陆速度惩罚（仅在接触时）
    if fcontact > 0.5:
        vel_pen = -w_land_vel * (nvx**2 + nvy**2)
    else:
        vel_pen = 0.0

    # 4. 姿态稳定惩罚（全程）
    att_penalty = -w_angle * (nangle**2) - w_angvel * (nangvel**2)

    # 5. 引擎使用惩罚（节省燃料）
    eng_pen = -engine_cost if action != 0 else 0.0

    total_reward = progress + landing_reward + vel_pen + att_penalty + eng_pen
    components = {
        "progress": progress,
        "landing_quality": landing_reward,
        "landing_velocity_penalty": vel_pen,
        "attitude_penalty": att_penalty,
        "engine_cost": eng_pen
    }
    return float(total_reward), components