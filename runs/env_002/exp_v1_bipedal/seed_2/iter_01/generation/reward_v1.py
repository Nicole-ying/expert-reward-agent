def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前向速度奖励
    forward_velocity = next_obs[2]  # 水平速度
    forward_reward = 1.0 * forward_velocity
    
    # 存活奖励：鼓励保持站立
    # 通过检查主体角度和角速度判断是否还站立
    hull_angle = abs(next_obs[0])  # 主体角度绝对值
    hull_angular_vel = abs(next_obs[1])  # 主体角速度绝对值
    # 如果角度太大或角速度太大，认为可能摔倒，不给存活奖励
    is_alive = (hull_angle < 1.0) and (hull_angular_vel < 2.0)
    alive_bonus = 0.5 if is_alive else 0.0
    
    # 稳定性惩罚：轻量约束，防止过度倾斜和高速旋转
    stability_penalty = -0.1 * (hull_angle + 0.5 * hull_angular_vel)
    
    # 总奖励
    total_reward = forward_reward + alive_bonus + stability_penalty
    
    components = {
        "forward_reward": forward_reward,
        "alive_bonus": alive_bonus,
        "stability_penalty": stability_penalty
    }
    
    return float(total_reward), components