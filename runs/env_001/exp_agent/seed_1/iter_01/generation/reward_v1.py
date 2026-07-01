def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward - 奖励每一步更接近目标
    # 计算当前位置到目标(0,0)的距离
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_delta_reward = 10.0 * progress_delta  # 权重10，鼓励持续接近
    
    # 稳定约束：stability_penalty - 惩罚高速、大姿态角和角速度
    # 使用next_obs的状态，因为动作影响后的状态更相关
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    
    # 速度惩罚：平方形式，对高速更敏感
    speed_penalty = -0.5 * (x_vel ** 2 + y_vel ** 2)
    # 姿态角惩罚：角度越大惩罚越大，使用平方
    angle_penalty = -0.3 * (body_angle ** 2)
    # 角速度惩罚：旋转越快惩罚越大
    angular_penalty = -0.2 * (angular_vel ** 2)
    
    stability_penalty = speed_penalty + angle_penalty + angular_penalty
    
    # 任务完成proxy：soft_landing_proxy - 当接近目标且稳定时给予小奖励
    # 条件：距离<0.5，速度<0.3，角度<0.2，角速度<0.2，且至少一个支撑接触
    near_target = next_dist < 0.5
    low_speed = (x_vel ** 2 + y_vel ** 2) ** 0.5 < 0.3
    stable_angle = abs(body_angle) < 0.2
    low_angular_vel = abs(angular_vel) < 0.2
    has_contact = (next_obs[6] > 0.5) or (next_obs[7] > 0.5)
    
    if near_target and low_speed and stable_angle and low_angular_vel and has_contact:
        soft_landing_bonus = 2.0
    else:
        soft_landing_bonus = 0.0
    
    # 动作代价：energy_penalty - 小权重惩罚使用引擎
    # 动作1、2、3都使用引擎，动作0不使用
    if action == 0:
        energy_penalty = 0.0
    else:
        energy_penalty = -0.1  # 小权重，避免agent不敢动
    
    # 总奖励
    total_reward = progress_delta_reward + stability_penalty + soft_landing_bonus + energy_penalty
    
    # 组件字典
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components