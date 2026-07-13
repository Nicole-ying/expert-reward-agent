def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前进速度奖励
    forward_velocity = next_obs[2]  # horizontal_velocity
    progress_reward = 2.0 * forward_velocity
    
    # 稳定性约束：惩罚身体倾斜和角速度
    hull_angle = next_obs[0]  # hull_angle
    hull_angular_velocity = next_obs[1]  # hull_angular_velocity
    stability_penalty = -0.5 * (hull_angle ** 2) - 0.1 * (hull_angular_velocity ** 2)
    
    # 步态质量奖励：鼓励交替接触地面
    leg1_contact = next_obs[12]  # leg1_contact
    leg2_contact = next_obs[13]  # leg2_contact
    # 当两条腿交替接触时给予奖励（异或逻辑）
    gait_quality = abs(leg1_contact - leg2_contact)  # 1.0 when exactly one leg contacts
    gait_bonus = 0.3 * gait_quality
    
    # 总奖励
    total_reward = progress_reward + stability_penalty + gait_bonus
    
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "gait_bonus": gait_bonus
    }
    
    return float(total_reward), components