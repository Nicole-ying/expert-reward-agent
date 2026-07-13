def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前进速度奖励
    # 使用 next_obs[2] 作为水平前进速度，鼓励 agent 向前移动
    forward_velocity = next_obs[2]
    # 限制速度范围，避免过大速度导致不稳定
    forward_velocity_clipped = max(-2.0, min(2.0, forward_velocity))
    # 只奖励正向速度（前进），负向速度（后退）不奖励
    forward_reward = 2.0 * max(0.0, forward_velocity_clipped)
    
    # 存活奖励：每步给予小额奖励，鼓励 agent 保持不倒
    # 通过检查 next_obs[0]（主体角度）是否在合理范围内来判断是否存活
    # 角度绝对值小于 1.0 弧度（约 57 度）视为存活
    hull_angle = next_obs[0]
    is_alive = abs(hull_angle) < 1.0
    alive_bonus = 0.5 if is_alive else 0.0
    
    # 稳定性惩罚：轻量约束，抑制过大角度和角速度
    # 使用 next_obs[0]（角度）和 next_obs[1]（角速度）
    angle_penalty = -0.1 * (hull_angle ** 2)
    angular_vel_penalty = -0.05 * (next_obs[1] ** 2)
    stability_penalty = angle_penalty + angular_vel_penalty
    
    # 总奖励
    total_reward = forward_reward + alive_bonus + stability_penalty
    
    # 组件字典
    components = {
        "progress_reward": forward_reward,
        "alive_bonus": alive_bonus,
        "stability_penalty": stability_penalty
    }
    
    return float(total_reward), components