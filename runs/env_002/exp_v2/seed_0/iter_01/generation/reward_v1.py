def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：progress_delta_reward ==========
    # 使用水平速度作为前进进度的代理
    # 奖励每一步水平速度的增加（即加速度为正）
    current_horizontal_vel = obs[2]
    next_horizontal_vel = next_obs[2]
    progress_delta = next_horizontal_vel - current_horizontal_vel
    progress_delta_reward = 2.0 * progress_delta  # 权重2.0，鼓励持续加速

    # ========== 稳定约束：stability_penalty ==========
    # 惩罚身体倾斜角度过大和角速度过大，防止摔倒
    hull_angle = next_obs[0]  # 身体倾斜角度
    hull_angular_vel = next_obs[1]  # 身体角速度
    # 使用平滑的指数惩罚，温度参数控制惩罚的陡峭程度
    temperature_angle = 0.5
    temperature_angular_vel = 0.3
    angle_penalty = -0.5 * (2.718281828 ** (abs(hull_angle) / temperature_angle) - 1.0)
    angular_vel_penalty = -0.3 * (2.718281828 ** (abs(hull_angular_vel) / temperature_angular_vel) - 1.0)
    stability_penalty = angle_penalty + angular_vel_penalty

    # ========== 效率约束：energy_penalty（小权重） ==========
    # 惩罚过大的关节力矩，鼓励节能
    action_magnitude = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = -0.01 * action_magnitude  # 权重很小，避免agent不敢动

    # ========== 任务完成proxy：soft_landing_proxy ==========
    # 当身体直立、速度适中、双脚着地时给予小奖励
    # 注意：不能直接使用contact作为success，必须组合条件
    leg1_contact = next_obs[12]
    leg2_contact = next_obs[13]
    both_contact = 1.0 if (leg1_contact > 0.5 and leg2_contact > 0.5) else 0.0
    # 直立条件：角度小于0.2弧度，角速度小于0.5
    is_upright = 1.0 if (abs(hull_angle) < 0.2 and abs(hull_angular_vel) < 0.5) else 0.0
    # 速度适中：水平速度在0.5到3.0之间
    is_good_speed = 1.0 if (0.5 < next_horizontal_vel < 3.0) else 0.0
    soft_landing_bonus = 0.5 * both_contact * is_upright * is_good_speed

    # ========== 总奖励 ==========
    total_reward = progress_delta_reward + stability_penalty + energy_penalty + soft_landing_bonus

    # ========== 组件字典 ==========
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "energy_penalty": energy_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "total_reward": total_reward
    }

    return float(total_reward), components