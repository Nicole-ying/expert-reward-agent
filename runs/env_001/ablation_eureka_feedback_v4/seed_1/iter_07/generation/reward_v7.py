def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    av = next_obs[5]
    lc = next_obs[6]
    rc = next_obs[7]

    dist = (x**2 + y**2) ** 0.5
    speed = (vx**2 + vy**2) ** 0.5

    # 接近奖励（线性距离惩罚，避免平方过陡）
    proximity_reward = -0.5 * dist

    # 速度惩罚（远处允许高速，近处要求减速）
    vel_gate = 1.0 / (1.0 + 5.0 * dist)
    velocity_penalty = -0.1 * (vx**2 + vy**2) * vel_gate

    # 姿态惩罚
    orientation_penalty = -0.5 * (angle**2) - 0.1 * (av**2)

    # 成功状态奖励：双接触 + 稳定（距离近、速度小、角度小）
    contact = lc * rc
    success_factor = (2.718281828 ** (-dist / 0.5)) * (2.718281828 ** (-speed / 0.3)) * (2.718281828 ** (-abs(angle) / 0.2))
    success_reward = 20.0 * contact * success_factor

    total_reward = proximity_reward + velocity_penalty + orientation_penalty + success_reward

    components = {
        'proximity_reward': proximity_reward,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'success_reward': success_reward
    }

    return float(total_reward), components