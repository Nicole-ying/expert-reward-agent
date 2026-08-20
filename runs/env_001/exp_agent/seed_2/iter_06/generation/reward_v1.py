def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 使用 next_obs 计算所有奖励组件，确保奖励反映动作后的状态
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. 主学习信号：密集距离奖励（负欧氏距离）
    distance = (x**2 + y**2) ** 0.5
    dist_reward = -distance

    # 2. 稳定约束：惩罚速度、姿态角和角速度
    stability_penalty = -0.1 * (abs(vx) + abs(vy)) \
                        - 0.2 * abs(angle) \
                        - 0.05 * abs(angular_vel)

    # 3. 任务完成近似信号：软着陆奖励（仅当多条件同时满足时激活）
    landing_proxy = 0.0
    if (abs(x) < 0.1 and abs(y) < 0.1 and 
        abs(vx) < 0.2 and abs(vy) < 0.2 and 
        abs(angle) < 0.1 and 
        left_contact == 1.0 and right_contact == 1.0):
        landing_proxy = 1.0

    total_reward = dist_reward + stability_penalty + landing_proxy

    components = {
        "dist_reward": dist_reward,
        "stability_penalty": stability_penalty,
        "landing_proxy": landing_proxy,
        "total_reward": total_reward
    }
    return float(total_reward), components