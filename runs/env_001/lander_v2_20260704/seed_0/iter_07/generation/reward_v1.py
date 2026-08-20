def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取 next_obs 的各个分量
    x_pos = next_obs[0]          # 水平位置
    y_pos = next_obs[1]          # 垂直位置
    x_vel = next_obs[2]          # 水平速度
    y_vel = next_obs[3]          # 垂直速度
    body_angle = next_obs[4]     # 身体角度
    left_contact = next_obs[6]   # 左支撑腿接触
    right_contact = next_obs[7]  # 右支撑腿接触

    # 主学习信号：非线性接近奖励，鼓励向目标靠近
    dist_to_target = (x_pos ** 2 + y_pos ** 2) ** 0.5
    proximity_reward = 1.0 / (1.0 + dist_to_target)

    # 稳定约束：抑制高速，防止剧烈运动
    velocity_penalty = -0.1 * (abs(x_vel) + abs(y_vel))

    # 姿态约束：抑制大幅倾斜，保证着陆姿态
    angle_penalty = -0.05 * abs(body_angle)

    # 任务完成近似信号：鼓励双腿接触着陆垫
    contact_reward = 0.3 * (left_contact + right_contact)

    total_reward = proximity_reward + velocity_penalty + angle_penalty + contact_reward

    components = {
        'proximity_reward': proximity_reward,
        'velocity_penalty': velocity_penalty,
        'angle_penalty': angle_penalty,
        'contact_reward': contact_reward
    }

    return float(total_reward), components