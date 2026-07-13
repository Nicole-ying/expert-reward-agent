def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前进速度奖励（bounded 形式，高速收益递减）
    forward_velocity = next_obs[2]  # horizontal_velocity
    # tanh 将速度映射到 (0,1)，系数 2.5 使典型速度 ~0.6 时奖励 ~1.3
    progress_reward = 2.5 * (forward_velocity / (1.0 + abs(forward_velocity)))
    
    # 存活奖励：鼓励 agent 保持存活状态
    alive_bonus = 0.5
    
    # 稳定性约束：惩罚身体倾斜和角速度
    hull_angle = next_obs[0]  # hull_angle
    hull_angular_velocity = next_obs[1]  # hull_angular_velocity
    stability_penalty = -0.5 * (hull_angle ** 2) - 0.1 * (hull_angular_velocity ** 2)
    
    # 总奖励
    total_reward = progress_reward + alive_bonus + stability_penalty
    
    components = {
        "progress_reward": progress_reward,
        "alive_bonus": alive_bonus,
        "stability_penalty": stability_penalty
    }
    
    return float(total_reward), components