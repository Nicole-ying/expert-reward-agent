def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 1. 主学习信号：基于水平速度的前进奖励 ==========
    # 使用 next_obs[2] (horizontal_velocity) 作为前进速度信号
    # 目标：鼓励机器人向前行走，速度越快奖励越高
    # 使用平滑的指数形式，避免梯度消失或爆炸
    temperature = 0.5  # 控制奖励曲线的陡峭程度
    forward_velocity = next_obs[2]  # 水平速度
    # 当速度为0时奖励为0，速度为正时奖励为正，速度为负时奖励为负
    forward_reward = 2.718281828 ** (forward_velocity / temperature) - 1.0
    # 限制范围避免极端值
    forward_reward = max(-10.0, min(10.0, forward_reward))
    
    # ========== 2. 稳定约束：姿态角惩罚 ==========
    # 使用 next_obs[0] (hull_angle) 惩罚身体倾斜
    # 目标：保持身体直立，避免摔倒
    hull_angle = next_obs[0]  # 主体角度，0为直立
    angle_penalty_weight = 0.5
    angle_penalty = -angle_penalty_weight * (hull_angle ** 2)
    
    # ========== 3. 稳定约束：角速度惩罚 ==========
    # 使用 next_obs[1] (hull_angular_velocity) 惩罚身体旋转
    # 目标：减少身体晃动，保持稳定
    angular_velocity = next_obs[1]
    angular_penalty_weight = 0.2
    angular_penalty = -angular_penalty_weight * (angular_velocity ** 2)
    
    # ========== 4. 效率约束：动作代价（小权重） ==========
    # 使用 action 计算能量消耗，鼓励高效行走
    energy_penalty_weight = 0.01
    energy_penalty = -energy_penalty_weight * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)
    
    # ========== 组合总奖励 ==========
    total_reward = forward_reward + angle_penalty + angular_penalty + energy_penalty
    
    # ========== 构建组件字典 ==========
    components = {
        "forward_reward": forward_reward,
        "angle_penalty": angle_penalty,
        "angular_penalty": angular_penalty,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components