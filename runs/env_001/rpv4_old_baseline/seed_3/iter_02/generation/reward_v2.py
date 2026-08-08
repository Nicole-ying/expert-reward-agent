def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取信号
    x, y = next_obs[0], next_obs[1]
    xv, yv = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    both_legs = left_contact * right_contact  # 1 如果双脚均接触

    # 当前距离与上一步距离
    dist_prev = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (x**2 + y**2) ** 0.5

    # ---- 主进度信号（approach_target）: improvement_delta ----
    progress = dist_prev - dist_next
    w_progress = 0.2

    # ---- 速度阻尼约束（velocity_damping）: 距离门控二次惩罚 ----
    gate = 1.0 / (1.0 + 0.1 * dist_next)          # 越近门控越大
    vel_penalty = xv**2 + yv**2
    w_vel = 0.5

    # ---- 姿态稳定约束（orientation_stabilization）: 二次惩罚 ----
    angle_penalty = angle**2
    angvel_penalty = ang_vel**2
    w_angle = 1.0
    w_angvel = 0.5

    # ---- 软着陆奖励（soft_landing）: 连续条件组合 ----
    prox = 1.0 / (1.0 + 10.0 * dist_next)                     # 靠近目标 ~1
    low_vel = 1.0 / (1.0 + 5.0 * abs(xv) + 5.0 * abs(yv))    # 速度低 ~1
    contact = both_legs                                       # 双脚已接触
    landing = prox * low_vel * contact
    w_land = 9.0  # 上调系数以增强着陆阶段的奖励

    # 组合
    total_reward = (
        w_progress * progress
        - w_vel * gate * vel_penalty
        - w_angle * angle_penalty
        - w_angvel * angvel_penalty
        + w_land * landing
    )

    components = {
        'progress': w_progress * progress,
        'velocity_damping': -w_vel * gate * vel_penalty,
        'orientation': -w_angle * angle_penalty - w_angvel * angvel_penalty,
        'soft_landing': w_land * landing,
    }

    return float(total_reward), components