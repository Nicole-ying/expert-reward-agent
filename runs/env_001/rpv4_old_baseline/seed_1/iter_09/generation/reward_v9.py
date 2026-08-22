def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解包状态
    x, y, vx, vy, angle, ang_vel, left, right = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nleft, nright = next_obs

    # 常用量
    dist = (x**2 + y**2)**0.5
    speed = (vx**2 + vy**2)**0.5
    next_dist = (nx**2 + ny**2)**0.5
    next_speed = (nvx**2 + nvy**2)**0.5

    # ---- 势能塑造：鼓励接近目标，同时惩罚高速 ----
    w_dist = 6.0                 # 适当加大距离权重
    w_speed = 3.0                # 略微降低速度惩罚，避免阻碍机动
    potential_cur = -(w_dist * dist + w_speed * speed)
    potential_next = -(w_dist * next_dist + w_speed * next_speed)
    shaping = potential_next - potential_cur

    # ---- 存活奖励：每步微小幅励，延长存活以增加着陆机会 ----
    survival_bonus = 0.05

    # ---- 角度惩罚：只在偏离竖直较大时生效，容忍小幅调整 ----
    angle_penalty = -2.0 * max(0.0, abs(nangle) - 0.25)

    # ---- 角速度惩罚：只在旋转过快时惩罚 ----
    angvel_penalty = -0.15 * max(0.0, abs(nang_vel) - 0.8)

    # ---- 燃料效率惩罚 ----
    if action == 2:           # 主引擎
        fuel_penalty = -0.2
    elif action in (1, 3):    # 姿态引擎
        fuel_penalty = -0.05
    else:                     # 无推力
        fuel_penalty = 0.0

    # ---- 地面接近危险速度惩罚（防止硬着陆/撞毁） ----
    # 当飞行器已接近地面（ny 很小）且快速下落时强力惩罚
    ground_danger_penalty = 0.0
    if ny < 0.4 and nvy < -0.35:
        ground_danger_penalty = -5.0 * (0.4 - ny) * (abs(nvy) ** 0.5)  # 越接近越危险

    # ---- 双腿接触持续奖励 ----
    contact_continuous = 0.0
    if nleft and nright:
        speed_factor = max(0.0, 1.0 - next_speed)          # 慢速着陆
        angle_factor = max(0.0, 1.0 - abs(nangle) / 0.3)
        contact_continuous = 2.0 * speed_factor * angle_factor   # 扩大份量

    # ---- 成功软着陆大奖励（提高至200分） ----
    success_bonus = 0.0
    if nleft and nright and (next_speed < 0.5) and (abs(nangle) < 0.2):
        success_bonus = 200.0

    # ---- 猛烈着陆惩罚 ----
    crash_penalty = 0.0
    if nleft and nright and ((next_speed > 2.0) or (abs(nangle) > 0.6)):
        crash_penalty = -30.0

    # 汇总
    total_reward = (shaping +
                    survival_bonus +
                    angle_penalty +
                    angvel_penalty +
                    fuel_penalty +
                    ground_danger_penalty +
                    contact_continuous +
                    success_bonus +
                    crash_penalty)

    components = {
        "shaping": shaping,
        "survival_bonus": survival_bonus,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "fuel_penalty": fuel_penalty,
        "ground_danger_penalty": ground_danger_penalty,
        "contact_continuous": contact_continuous,
        "success_bonus": success_bonus,
        "crash_penalty": crash_penalty
    }

    return float(total_reward), components