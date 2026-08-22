def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前观测
    x, y = obs[0], obs[1]
    # 下一观测
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    nangle = abs(next_obs[4])          # body_angle
    nangvel = abs(next_obs[5])         # angular_velocity
    nleft = next_obs[6]
    nright = next_obs[7]

    # 主进展信号：与目标着陆垫的距离变化
    d_current = (x**2 + y**2)**0.5
    d_next = (nx**2 + ny**2)**0.5 + 1e-8   # 避免除零
    progress = d_current - d_next          # >0 表示接近目标

    # soft health gate —— 在姿态、速度、角速度恶化时衰减正向进展
    # 使用指数衰减，温度系数根据任务特性设定
    angle_gate   = 2.718281828**(-2.0 * nangle**2)                       # 身体倾角过大时衰减
    v_sq = nvx**2 + nvy**2
    vel_gate     = 2.718281828**(-1.0 * v_sq / (1.0 + d_next))           # 高速时衰减，距离越近越严格
    angvel_gate  = 2.718281828**(-3.0 * nangvel)                         # 角速度过大时衰减

    gate = angle_gate * vel_gate * angvel_gate

    # 仅对正向进展施加 gate，负进展（远离）保留完整惩罚
    if progress > 0.0:
        progress *= gate

    # 任务完成近似信号：双腿同时接触着陆垫时给予持续性小奖励
    contact_reward = 0.2 if (nleft > 0.5 and nright > 0.5) else 0.0

    total_reward = progress + contact_reward
    components = {
        "progress": progress,
        "contact": contact_reward
    }
    return float(total_reward), components