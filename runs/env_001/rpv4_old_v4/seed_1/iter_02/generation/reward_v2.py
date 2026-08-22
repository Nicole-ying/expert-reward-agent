def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v2 — 语义修改：着陆反馈从接触门控改为高度门控。

    变更：
      - 删除 soft_landing_bonus（contact-gated 三重乘积，active_rate=0.3%）
      - 新增 landing_speed：近地速度惩罚，高度门控
      - 新增 landing_quality：近地品质奖励，鼓励低速+水平姿态
      - proximity_progress 系数 30→20，为着陆信号留权重空间
    """

    # ── 组件 A：接近进度（改善量信号） ──
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 20.0 * (old_dist - new_dist)  # 靠近>0, 远离<0

    # ── 组件 B：姿态惩罚（全程） ──
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.5 * (angle ** 2) - 0.1 * (angular_vel ** 2)

    # ── 高度门控因子 ──
    # y_pos 从 0.5→0 时 gate 从 0→1 线性 ramp
    y_pos = next_obs[1]
    height_gate = max(0.0, min(1.0, (0.5 - y_pos) / 0.5))

    # ── 组件 C：近地速度惩罚 ──
    # 高度越低，速度惩罚越强；高空不约束速度
    speed_penalty = -0.5 * (next_obs[2] ** 2 + next_obs[3] ** 2)
    landing_speed = height_gate * speed_penalty

    # ── 组件 D：近地着陆品质奖励 ──
    # 鼓励低速 + 水平姿态，仅在近地区域生效
    speed_mag = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    speed_quality = 2.718281828 ** (-4.0 * speed_mag)      # 速度越低越接近 1
    angle_quality = 2.718281828 ** (-4.0 * abs(angle))     # 角度越小越接近 1
    landing_quality = 1.5 * height_gate * speed_quality * angle_quality

    # ── 总奖励 ──
    total_reward = proximity_progress + attitude_penalty + landing_speed + landing_quality

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "landing_speed": landing_speed,
        "landing_quality": landing_quality
    }

    return float(total_reward), components