import numpy as np

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    奖励函数 v1：基于距离的密集奖励 + 速度惩罚 + 姿态稳定性约束
    """
    # 提取当前状态
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    # 提取下一状态
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_angular_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # 1. 距离奖励：鼓励接近目标（目标在原点）
    current_distance = np.sqrt(x_pos**2 + y_pos**2)
    next_distance = np.sqrt(next_x_pos**2 + next_y_pos**2)
    distance_reward = -current_distance  # 距离越近奖励越高
    
    # 2. 增量进步奖励：鼓励每一步都更接近目标
    progress_reward = current_distance - next_distance  # 距离减小为正奖励
    
    # 3. 速度惩罚：鼓励减速，尤其是接近目标时
    current_speed = np.sqrt(x_vel**2 + y_vel**2)
    next_speed = np.sqrt(next_x_vel**2 + next_y_vel**2)
    speed_penalty = -0.1 * next_speed  # 速度越快惩罚越大
    
    # 4. 姿态稳定性惩罚：鼓励保持水平姿态
    angle_penalty = -0.05 * (next_body_angle**2 + 0.1 * next_angular_vel**2)
    
    # 5. 接触奖励：鼓励着陆在平台上
    contact_reward = 0.0
    if next_left_contact > 0.5 and next_right_contact > 0.5:
        # 两个支撑点都接触，说明已着陆
        contact_reward = 1.0
    
    # 6. 动作惩罚：鼓励少用引擎
    action_penalty = 0.0
    if action == 1 or action == 2 or action == 3:
        action_penalty = -0.02  # 使用任何引擎都轻微惩罚
    
    # 组合奖励
    total_reward = (
        distance_reward * 0.3 +
        progress_reward * 0.5 +
        speed_penalty +
        angle_penalty +
        contact_reward * 2.0 +
        action_penalty
    )
    
    # 记录奖励项用于诊断
    info["reward_terms"] = {
        "distance_reward": distance_reward * 0.3,
        "progress_reward": progress_reward * 0.5,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "contact_reward": contact_reward * 2.0,
        "action_penalty": action_penalty,
        "total_reward": total_reward
    }
    
    return total_reward