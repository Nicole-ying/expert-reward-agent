def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前步与下一步的相对目标距离
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 非线性接近度：有界且近目标时梯度更强
    prox_current = 1.0 / (1.0 + dist_current)
    prox_next = 1.0 / (1.0 + dist_next)

    # 状态改善量：奖励靠近，惩罚远离，停留则趋零（系数使整回合总量与contact可比）
    proximity_improvement = 80.0 * (prox_next - prox_current)

    # 稳定约束：抑制高速
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    velocity_penalty = -0.1 * (abs(x_vel) + abs(y_vel))

    # 姿态约束：抑制大幅倾斜
    angle_penalty = -0.05 * abs(next_obs[4])

    # 着陆信号：奖励双腿触地
    contact_reward = 0.3 * (next_obs[6] + next_obs[7])

    total_reward = proximity_improvement + velocity_penalty + angle_penalty + contact_reward

    components = {
        'proximity_improvement': proximity_improvement,
        'velocity_penalty': velocity_penalty,
        'angle_penalty': angle_penalty,
        'contact_reward': contact_reward
    }

    return float(total_reward), components