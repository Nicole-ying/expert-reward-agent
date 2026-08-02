def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y, vx, vy, angle, angvel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, nangle, nangvel, nleft, nright = next_obs

    # ----- 1. 主进展信号：向中心靠近 -----
    dist = (x**2 + y**2)**0.5 + 1e-8
    next_dist = (nx**2 + ny**2)**0.5 + 1e-8
    progress = 5.0 * (dist - next_dist)   # 正奖励=靠近中心

    # ----- 2. 完成信号（乘积式 joint condition proxy）-----
    # 每个因子在条件未满足时为 0，满足时线性趋近 1
    proximity      = max(0.0, 1.0 - next_dist / 0.3)          # 距离 <0.3 时才有正贡献
    velocity_ok    = max(0.0, 1.0 - (abs(nvx) + abs(nvy)) / 0.5)  # 联合线速度 <0.5 时非零
    angle_ok       = max(0.0, 1.0 - abs(nangle) / 0.2)        # 倾角 <0.2 rad
    angvel_ok      = max(0.0, 1.0 - abs(nangvel) / 0.3)       # 角速度 <0.3
    contact_ok     = (nleft + nright) / 2.0                   # 双脚接触=1, 单脚=0.5, 无=0

    completion = proximity * velocity_ok * angle_ok * angvel_ok * contact_ok
    completion_bonus = 15.0 * completion   # 只有全部条件同时接近满足时才大

    # ----- 3. 安全约束（hinge 惩罚，仅在超出阈值时生效）-----
    speed_penalty  = -0.3 * (max(0.0, abs(nvx) - 0.5) + max(0.0, abs(nvy) - 0.5))
    angle_penalty  = -0.6 * max(0.0, abs(nangle) - 0.2)
    angvel_penalty = -0.2 * max(0.0, abs(nangvel) - 0.5)

    total_reward = progress + completion_bonus + speed_penalty + angle_penalty + angvel_penalty

    components = {
        'progress': progress,
        'completion_bonus': completion_bonus,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'angvel_penalty': angvel_penalty
    }

    return float(total_reward), components