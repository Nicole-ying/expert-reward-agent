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

    w_progress = 20.0
    w_landing = 0.3
    w_land_vel = 0.2            # 重新激活，改为 hinge 式
    w_angle = 0.5
    w_angvel = 0.5
    engine_cost = 0.02

    dist = (x**2 + y**2) ** 0.5
    ndist = (nx**2 + ny**2) ** 0.5

    # 1. 距离改进
    progress = w_progress * (dist - ndist)

    # 2. 着陆质量软信号 —— 仅在双腿同时接触时激活
    if nleft_contact > 0.5 and nright_contact > 0.5:
        altitude_factor = max(0.0, 1.0 - abs(ny) / 0.5)
        align_factor    = max(0.0, 1.0 - abs(nx) / 0.5)
        vx_factor       = max(0.0, 1.0 - abs(nvx) / 0.3)
        vy_factor       = max(0.0, 1.0 - abs(nvy) / 0.5)
        angle_factor    = max(0.0, 1.0 - abs(nangle) / 0.2)
        product = (altitude_factor * align_factor * vx_factor *
                   vy_factor * angle_factor)
        if product > 0.0:
            landing_quality = w_landing * (product ** (1.0 / 5.0))
        else:
            landing_quality = 0.0
    else:
        landing_quality = 0.0

    # 3. 着陆速度惩罚 (hinge) —— 仅双腿接触时激活，惩罚超出安全阈值的部分
    if nleft_contact > 0.5 and nright_contact > 0.5:
        excess_vx = max(0.0, abs(nvx) - 0.2)
        excess_vy = max(0.0, abs(nvy) - 0.3)
        vel_pen = -w_land_vel * (excess_vx + excess_vy)
    else:
        vel_pen = 0.0

    # 4. 姿态稳定惩罚（全程）
    att_penalty = -w_angle * (nangle**2) - w_angvel * (nangvel**2)

    # 5. 引擎使用惩罚
    eng_pen = -engine_cost if action != 0 else 0.0

    total_reward = progress + landing_quality + vel_pen + att_penalty + eng_pen
    components = {
        "progress": progress,
        "landing_quality": landing_quality,
        "landing_velocity_penalty": vel_pen,
        "attitude_penalty": att_penalty,
        "engine_cost": eng_pen
    }
    return float(total_reward), components