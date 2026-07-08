def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ======== 提取观测 ========
    x_t, y_t = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    w_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ======== 1. 主学习信号：进步奖励 (progress_delta) ========
    dist_t = (x_t ** 2 + y_t ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    progress = dist_t - dist_next   # 正值为接近目标
    progress_reward = 0.5 * progress   # 缩放因子，使单步典型值在 0.0~0.1 左右

    # ======== 2. 软着陆近似奖励 (soft_landing_proxy) ========
    # 条件：接近目标、低速、角度小、双腿都已接触
    near_target = dist_next < 0.1
    low_speed = (abs(vx_next) + abs(vy_next)) < 0.1
    stable_angle = abs(angle_next) < 0.1
    both_contact = (left_contact > 0.5) and (right_contact > 0.5)
    
    soft_landing_reward = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_reward = 1.0   # 单步事件型奖励，鼓励稳定着陆

    # ======== 3. 轻量稳定性惩罚 (stability_penalty) ========
    # 小权重抑制高速、大角度、高角速度，但不压制主学习信号
    stability_penalty = -0.002 * (abs(vx_next) + abs(vy_next)) \
                        - 0.002 * abs(angle_next) \
                        - 0.001 * abs(w_next)

    # ======== 总奖励 ========
    total_reward = progress_reward + soft_landing_reward + stability_penalty

    # ======== 组件字典 ========
    components = {
        'progress_reward': progress_reward,
        'soft_landing_reward': soft_landing_reward,
        'stability_penalty': stability_penalty,
        'total_reward': total_reward
    }

    return float(total_reward), components