def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y, vx, vy, angle, angvel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, nangle, nangvel, nleft, nright = next_obs

    # 到目标中心距离
    dist = (x**2 + y**2)**0.5 + 1e-8
    next_dist = (nx**2 + ny**2)**0.5 + 1e-8

    # 基准进展信号（delta）
    progress = 5.0 * (dist - next_dist)

    # 接触奖励（引导脚触地）
    contact_reward = 0.2 * (nleft + nright)

    # 完成因子（各子条件连续映射到[0,1]）
    proximity_factor = max(0.0, 1.0 - next_dist / 0.3)            # 距中心<0.3
    velocity_factor  = max(0.0, 1.0 - (abs(nvx) + abs(nvy)) / 0.3)  # 合速度<0.3
    angle_factor     = max(0.0, 1.0 - abs(nangle) / 0.15)         # 倾角<0.15 rad
    angvel_factor    = max(0.0, 1.0 - abs(nangvel) / 0.2)         # 角速度<0.2 rad/s
    contact_factor   = (nleft + nright) / 2.0                     # 双脚接触程度

    # min‑joint completion：只有最差条件改善总分才提高
    completion = 10.0 * min(proximity_factor, velocity_factor, angle_factor, angvel_factor, contact_factor)

    # 安全阈值惩罚（降低阈值使约束可感知）
    speed_penalty    = -0.5 * (max(0.0, abs(nvx) - 0.4) + max(0.0, abs(nvy) - 0.4))
    angle_penalty    = -1.0 * max(0.0, abs(nangle) - 0.15)
    angvel_penalty   = -0.3 * max(0.0, abs(nangvel) - 0.3)
    boundary_penalty = -2.0 * max(0.0, abs(nx) - 0.8)   # 水平出界预警

    total_reward = (progress + contact_reward + completion +
                    speed_penalty + angle_penalty + angvel_penalty + boundary_penalty)

    components = {
        'progress': progress,
        'contact_reward': contact_reward,
        'completion': completion,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'angvel_penalty': angvel_penalty,
        'boundary_penalty': boundary_penalty
    }

    return float(total_reward), components