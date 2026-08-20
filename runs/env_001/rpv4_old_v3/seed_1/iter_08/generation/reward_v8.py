def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前观测
    x, y = obs[0], obs[1]
    # 下一观测
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    nangle = abs(next_obs[4])
    nangvel = abs(next_obs[5])
    nleft = next_obs[6]
    nright = next_obs[7]

    # 主进展信号：与目标着陆垫的距离变化
    d_current = (x**2 + y**2)**0.5
    d_next = (nx**2 + ny**2)**0.5 + 1e-8
    progress = d_current - d_next

    # health gate —— 衰减正向进展（仅距离改善方向）
    angle_gate  = 2.718281828**(-2.0 * nangle**2)
    v_sq = nvx**2 + nvy**2
    vel_gate    = 2.718281828**(-1.0 * v_sq / (1.0 + d_next))
    angvel_gate = 2.718281828**(-3.0 * nangvel)

    gate = angle_gate * vel_gate * angvel_gate
    if progress > 0.0:
        progress *= gate

    # 接触+稳定性奖励：双腿接触时根据姿态和速度稳定性给予连续奖励
    contact_reward = 0.0
    if nleft > 0.5 and nright > 0.5:
        # 稳定性因子：倾斜、线速度、角速度越低，奖励越接近 1.0
        stability = (
            2.718281828**(-2.0 * nangle) *      # 倾斜惩罚
            2.718281828**(-(v_sq)) *             # 线速度惩罚
            2.718281828**(-3.0 * nangvel)        # 角速度惩罚
        )
        contact_reward = 0.2 * stability

    total_reward = progress + contact_reward
    components = {
        "progress": progress,
        "contact": contact_reward
    }
    return float(total_reward), components