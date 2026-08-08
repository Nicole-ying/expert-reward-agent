def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解包观测 (环境卡声明的 8 维)
    x, y, vx, vy, angle, ang_vel, left, right = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nleft, nright = next_obs

    # 计算当前与下一步的距离和速度幅值
    dist = (x**2 + y**2)**0.5
    speed = (vx**2 + vy**2)**0.5
    next_dist = (nx**2 + ny**2)**0.5
    next_speed = (nvx**2 + nvy**2)**0.5

    # ---- 主学习信号：potential shaping 鼓励靠近目标垫并减速 ----
    w_dist = 2.0
    w_speed = 1.0
    potential_cur = -(w_dist * dist + w_speed * speed)
    potential_next = -(w_dist * next_dist + w_speed * next_speed)
    shaping_reward = potential_next - potential_cur  # 正值 = 靠近/减速

    # ---- 稳定约束 1：姿态惩罚 ----
    w_angle = 0.5
    angle_penalty = -w_angle * (angle**2)

    # ---- 稳定约束 2：速度惩罚 ----
    w_vel = 0.1
    vel_penalty = -w_vel * (speed**2)

    # ---- 效率：离散动作燃料惩罚 ----
    if action == 2:           # 主发动机
        fuel_penalty = -0.15
    elif action in (1, 3):    # 左/右姿态引擎
        fuel_penalty = -0.02
    else:                     # 无推力
        fuel_penalty = 0.0

    # ---- 任务完成近似信号：双腿接触 + 位置/速度近零的软乘积奖励 ----
    w_contact = 2.0
    pos_thresh = 1.0
    speed_thresh = 0.5
    proximity = max(0.0, 1.0 - (abs(nx) + abs(ny)) / pos_thresh)
    velocity_factor = max(0.0, 1.0 - next_speed / speed_thresh)
    contact_score = nleft * nright * proximity * velocity_factor
    contact_reward = w_contact * contact_score

    total_reward = (
        shaping_reward
        + angle_penalty
        + vel_penalty
        + fuel_penalty
        + contact_reward
    )

    components = {
        "shaping_reward": shaping_reward,
        "angle_penalty": angle_penalty,
        "vel_penalty": vel_penalty,
        "fuel_penalty": fuel_penalty,
        "contact_reward": contact_reward,
    }

    return float(total_reward), components