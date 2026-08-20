def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取当前观测
    x, y = obs[0], obs[1]

    # 提取下一时刻观测
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_angvel = next_obs[5]
    nl_contact = next_obs[6]
    nr_contact = next_obs[7]

    # 超参数
    PROGRESS_SCALE = 1.0
    CLIP_MAX = 1.0
    LANDING_SCALE = 2.0
    DIST_THRESH = 1.0
    VEL_THRESH = 0.3
    ANGLE_THRESH = 0.05
    ANGVEL_THRESH = 0.05

    # 主学习信号：向目标前进的稠密奖励
    dist_prev = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    delta_dist = dist_prev - dist_next
    delta_dist_clipped = max(-CLIP_MAX, min(CLIP_MAX, delta_dist))
    progress_reward = PROGRESS_SCALE * delta_dist_clipped

    # 近距门控因子
    proximity = max(0.0, 1.0 - dist_next / DIST_THRESH)

    # 稳定性子条件（不含contact）
    speed = (nvx**2 + nvy**2) ** 0.5
    speed_quality = max(0.0, 1.0 - speed / VEL_THRESH)
    angle_quality = max(0.0, 1.0 - abs(n_angle) / ANGLE_THRESH)
    angvel_quality = max(0.0, 1.0 - abs(n_angvel) / ANGVEL_THRESH)

    # contact作为软性乘法门控：无接触时奖励降至20%，双足着地时完整释放
    contact_quality = 0.5 * (nl_contact + nr_contact)
    contact_gate = 0.2 + 0.8 * contact_quality

    # 稳定性均值 + contact门控：联合满足结构，contact不可被速度和姿态完全补偿
    stability_avg = (speed_quality + angle_quality + angvel_quality) / 3.0
    soft_landing_proxy = LANDING_SCALE * proximity * stability_avg * contact_gate

    total_reward = progress_reward + soft_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "soft_landing_proxy": soft_landing_proxy
    }

    return float(total_reward), components