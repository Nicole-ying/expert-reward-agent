def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Env_003 生存平衡任务 reward_v3。

    诊断：v1/v2 的纯负惩罚始终与 original_env_reward (+1.0) 符号相反，
    agent 学到"最不差"而非"最好"。v1 二次形态的平衡质量 (-0.059) 优于 v2 绝对值 (-0.189)。

    改动 (v2→v3)：回到 v1 二次惩罚系数，并增加正基线 +1.0。
    - 良好平衡时 reward≈+1.0，与生存奖励同向对齐，提供正向强化。
    - 偏离增大时 reward 平滑下降，极端情况可转负，梯度连续。
    - 保持 v1 验证过的系数：w_angle=5.0, w_angular_vel=0.5, w_pos=0.1。
    """
    base_pos = next_obs[0]
    pole_angle = next_obs[2]
    pole_angular_vel = next_obs[3]

    w_angle = 5.0
    w_angular_vel = 0.5
    w_pos = 0.1

    # 偏离惩罚：二次形态（v1 验证，精细平衡优于绝对值）
    penalty = (
        w_angle * (pole_angle ** 2) +
        w_angular_vel * (pole_angular_vel ** 2) +
        w_pos * (base_pos ** 2)
    )

    # 正基线奖励：完美平衡时≈1.0，对齐生存奖励方向
    progress_reward = 1.0 - penalty

    total_reward = progress_reward
    components = {"progress_reward": progress_reward}

    return float(total_reward), components