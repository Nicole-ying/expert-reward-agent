def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- 配置参数 ---
    # 主学习信号权重
    w_progress = 1.0

    # 稳定性惩罚权重
    w_speed = 0.01
    w_angle = 0.01
    w_angvel = 0.001

    # 软着陆代理奖励权重及其阈值
    w_proxy = 0.2
    dist_threshold = 0.2      # 距离目标平台足够近
    speed_threshold = 0.1     # 速度足够低
    angle_threshold = 0.2     # 姿态接近水平
    # 接触标志：0.0 未接触，1.0 接触

    # --- 1. 进度差分奖励 (主学习信号) ---
    # 当前位置到目标 (0,0) 的距离
    d_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    # 下一位置到目标的距离
    d_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = d_current - d_next

    # --- 2. 稳定性惩罚 (稳定约束) ---
    # 线速度大小
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    # 机体倾角绝对值
    angle_abs = abs(next_obs[4])
    # 角速度绝对值
    angvel_abs = abs(next_obs[5])

    stability_penalty = - (w_speed * speed + w_angle * angle_abs + w_angvel * angvel_abs)

    # --- 3. 软着陆代理奖励 (任务完成近似) ---
    # 条件：靠近目标、低速、姿态稳定、双脚均已接触
    near_target = d_next < dist_threshold
    low_speed = speed < speed_threshold
    stable_angle = angle_abs < angle_threshold
    both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)

    soft_proxy = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_proxy = w_proxy * 1.0

    # --- 总奖励 ---
    total_reward = progress + stability_penalty + soft_proxy

    # 组件字典
    components = {
        'progress': progress,
        'stability_penalty': stability_penalty,
        'soft_proxy': soft_proxy,
        'total_reward': total_reward
    }

    return float(total_reward), components