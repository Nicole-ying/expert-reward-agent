def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Env_003 生存平衡任务 reward_v1。
    使用连续负惩罚引导摆杆保持竖直并使底座接近轨道中心。
    """
    # 提取下一步观察
    base_pos = next_obs[0]       # 底座水平位置
    pole_angle = next_obs[2]     # 杆偏角 (rad)
    pole_angular_vel = next_obs[3]  # 杆角速度

    # 惩罚系数
    w_angle = 5.0
    w_angular_vel = 0.5
    w_pos = 0.1

    # 核心学习信号：偏离直立和中心的二次惩罚
    progress_reward = -(
        w_angle * (pole_angle ** 2) +
        w_angular_vel * (pole_angular_vel ** 2) +
        w_pos * (base_pos ** 2)
    )

    total_reward = progress_reward
    components = {"progress_reward": progress_reward}

    return float(total_reward), components