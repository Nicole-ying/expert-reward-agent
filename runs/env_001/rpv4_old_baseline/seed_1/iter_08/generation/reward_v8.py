def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解包状态
    x, y, vx, vy, angle, ang_vel, left, right = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nleft, nright = next_obs

    # 常用量
    dist = (x**2 + y**2)**0.5
    speed = (vx**2 + vy**2)**0.5
    next_dist = (nx**2 + ny**2)**0.5
    next_speed = (nvx**2 + nvy**2)**0.5

    # ---- 势能塑造：鼓励接近目标、保持低速 ----
    w_dist = 5.0
    w_speed = 5.0          # 适度惩罚速度，不阻碍必要移动
    potential_cur = -(w_dist * dist + w_speed * speed)
    potential_next = -(w_dist * next_dist + w_speed * next_speed)
    shaping = potential_next - potential_cur

    # ---- 姿态与角速度惩罚 ----
    angle_penalty = -0.5 * abs(nangle)       # 鼓励竖直
    angvel_penalty = -0.05 * abs(nang_vel)   # 抑制旋转

    # ---- 燃料效率惩罚 ----
    if action == 2:           # 主引擎
        fuel_penalty = -0.15
    elif action in (1, 3):    # 姿态引擎
        fuel_penalty = -0.02
    else:                     # 无推力
        fuel_penalty = 0.0

    # ---- 步数惩罚，推动尽快完成任务 ----
    step_penalty = -0.02

    # ---- 双腿接触持续奖励（鼓励稳定软着陆） ----
    contact_continuous = 0.0
    if nleft and nright:
        speed_factor = max(0.0, 1.0 - next_speed)          # 速度越慢越好，线性衰减到0
        angle_factor = max(0.0, 1.0 - abs(nangle) / 0.3)  # 倾角小于0.3 rad 时线性
        contact_continuous = 1.0 * speed_factor * angle_factor

    # ---- 成功软着陆大奖励 ----
    contact_success = (nleft and nright and
                       (next_speed < 0.5) and
                       (abs(nangle) < 0.2))
    success_bonus = 100.0 if contact_success else 0.0

    # ---- 猛烈着陆惩罚 ----
    crash_condition = nleft and nright and ((next_speed > 2.0) or (abs(nangle) > 0.5))
    crash_penalty = -20.0 if crash_condition else 0.0

    # 汇总
    total_reward = (shaping +
                    angle_penalty +
                    angvel_penalty +
                    fuel_penalty +
                    step_penalty +
                    contact_continuous +
                    success_bonus +
                    crash_penalty)

    components = {
        "shaping": shaping,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "fuel_penalty": fuel_penalty,
        "step_penalty": step_penalty,
        "contact_continuous": contact_continuous,
        "success_bonus": success_bonus,
        "crash_penalty": crash_penalty
    }

    return float(total_reward), components