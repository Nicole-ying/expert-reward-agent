def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 计算当前和下一步到目标中心的欧氏距离
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    
    # 1. 距离缩短奖励（靠近目标为正）
    approach_delta = current_dist - next_dist
    approach_reward = 8.0 * approach_delta
    
    # 2. 成功着陆软奖励：联合条件代理（几何平均防塌缩）
    f_dist = max(0.0, 1.0 - next_dist / 0.3)               # 距离因子
    speed_mag = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    f_speed = max(0.0, 1.0 - speed_mag / 0.5)              # 速度因子
    f_angle = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)      # 姿态角因子
    f_contact = (next_obs[6] + next_obs[7]) / 2.0          # 接触因子
    success_proxy = (f_dist * f_speed * f_angle * f_contact) ** 0.25
    success_reward = 3.0 * success_proxy
    
    # 3. 燃料消耗惩罚（离散动作：非零动作即惩罚）
    fuel_penalty = -0.05 if action != 0 else 0.0
    
    total = approach_reward + success_reward + fuel_penalty
    
    components = {
        'approach_delta': float(approach_delta),
        'success_proxy': float(success_proxy),
        'fuel_penalty': float(fuel_penalty)
    }
    return (float(total), components)