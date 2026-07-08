def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Env_003 生存平衡任务 reward_v2。
    
    改动 (v1→v2)：将二次惩罚改为绝对值惩罚。
    - 二次惩罚在接近零时梯度消失，绝对值提供恒定梯度，便于精细平衡。
    - 系数重新标定以在终止阈值附近保持与 v1 相同的惩罚量级。
    """
    # 提取下一步观察
    base_pos = next_obs[0]            # 底座水平位置
    pole_angle = next_obs[2]          # 杆偏角 (rad)
    pole_angular_vel = next_obs[3]    # 杆角速度

    # 惩罚系数（已针对绝对值形态重新标定）
    w_angle = 1.0
    w_angular_vel = 0.5
    w_pos = 0.25

    # 核心学习信号：偏离直立和中心的绝对值惩罚
    progress_reward = -(
        w_angle * abs(pole_angle) +
        w_angular_vel * abs(pole_angular_vel) +
        w_pos * abs(base_pos)
    )

    total_reward = progress_reward
    components = {"progress_reward": progress_reward}

    return float(total_reward), components