def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：前进速度奖励 ==========
    # 使用水平速度(obs[2])作为前进驱动力
    # 鼓励agent向前移动，这是任务的核心目标
    forward_velocity = next_obs[2]  # 使用next_obs避免延迟
    fwd_scale = 2.0
    progress_reward = fwd_scale * forward_velocity
    
    # ========== 存活奖励：鼓励持续行走 ==========
    # 通过主体角度(obs[0])和角速度(obs[1])判断是否还站立
    # 当主体角度过大或角速度过高时，说明即将摔倒，减少存活奖励
    hull_angle = abs(next_obs[0])  # 主体偏离竖直方向的角度
    hull_angular_vel = abs(next_obs[1])  # 主体角速度绝对值
    
    # 存活条件：角度小于0.5弧度(~28度)且角速度小于2.0
    alive_condition = (hull_angle < 0.5) and (hull_angular_vel < 2.0)
    alive_bonus = 0.5 if alive_condition else 0.0
    
    # ========== 稳定性约束：轻量惩罚 ==========
    # 惩罚过大的主体角度和角速度，防止摔倒
    # 使用连续函数，避免二值惩罚导致的梯度消失
    angle_penalty_scale = 0.5
    angular_vel_penalty_scale = 0.3
    
    angle_penalty = angle_penalty_scale * (hull_angle ** 2)
    angular_vel_penalty = angular_vel_penalty_scale * (hull_angular_vel ** 2)
    stability_penalty = -(angle_penalty + angular_vel_penalty)
    
    # ========== 总奖励 ==========
    total_reward = progress_reward + alive_bonus + stability_penalty
    
    # ========== 组件记录 ==========
    components = {
        "progress_reward": progress_reward,
        "alive_bonus": alive_bonus,
        "stability_penalty": stability_penalty
    }
    
    return float(total_reward), components