def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 读取当前状态
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 从 obs 读取上一步位置，计算距离变化
    px = obs[0]
    py = obs[1]

    prev_dist = (px**2 + py**2) ** 0.5
    curr_dist = (x**2 + y**2) ** 0.5

    # === 1. 主学习信号：进度增量（向目标靠近为正） ===
    # 这是 potential-based shaping 的特例：Φ = -distance, γ = 1
    # 正奖励 = 距离减小，天然零均值，提供清晰的梯度方向
    progress = prev_dist - curr_dist

    # === 2. 连续着陆质量信号（始终激活，峰值在着陆条件） ===
    # 三个 bounded 因子：proximity, speed, upright，都是 1/(1+kx) 形式
    # 自动 bounded 在 [0,1]，无需手动调尺度

    # 接近度：距离越近越高，k=5 使 distance=1 时约为 0.17
    proximity = 1.0 / (1.0 + 5.0 * curr_dist)

    # 速度因子：越慢越高，k=3 使 speed=1 时约为 0.25
    speed_val = (vx**2 + vy**2) ** 0.5
    speed_factor = 1.0 / (1.0 + 3.0 * speed_val)

    # 姿态因子：越正越高，k=3 使 angle=0.3rad 时约为 0.53
    upright_factor = 1.0 / (1.0 + 3.0 * abs(angle))

    # 接触加成：有腿部接触时额外奖励（0.5 到 1.0 之间）
    contact_bonus = 0.5 + 0.5 * (left_contact + right_contact) / 2.0

    # 连续乘积：每个因子 ∈ (0,1]，乘积提供密集梯度
    landing_quality = proximity * speed_factor * upright_factor * contact_bonus

    # === 3. 稳定性惩罚：距离门控，远处不罚 ===
    # 只在目标附近（<2 单位）施加弱稳定性约束
    gate = max(0.0, 1.0 - curr_dist / 2.0)
    stability_penalty = -gate * 0.02 * (abs(vx) + abs(vy) + abs(angular_vel))

    # === 组合 ===
    # progress: 方向性梯度，均值接近 0
    # landing_quality: 密集正信号，引导精细着陆行为
    # stability_penalty: 弱背景约束，仅在近端生效
    total_reward = progress + 1.0 * landing_quality + stability_penalty

    components = {
        "progress": progress,
        "landing_quality": landing_quality,
        "stability_penalty": stability_penalty,
        "total_reward": total_reward
    }
    return float(total_reward), components