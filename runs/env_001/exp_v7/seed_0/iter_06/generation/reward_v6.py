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
    progress_reward = 0.5 * progress

    # ======== 2. 连续软着陆奖励 (soft_landing_continuous) ========
    # 从二值事件改为连续乘积——每个因子在 [0,1]，提供平滑梯度
    # 距离因子：dist_next=0 时为 1，dist_next>=0.3 时为 0
    proximity = max(0.0, 1.0 - dist_next / 0.3)
    # 速度因子：总速度=0 时为 1，总速度>=0.3 时为 0
    speed_norm = abs(vx_next) + abs(vy_next)
    speed_factor = max(0.0, 1.0 - speed_norm / 0.3)
    # 角度因子：angle=0 时为 1，|angle|>=0.2 时为 0
    angle_factor = max(0.0, 1.0 - abs(angle_next) / 0.2)
    # 接触因子：鼓励双腿同时接触，单腿=0.5，双腿=1.0
    contact_factor = (left_contact + right_contact) / 2.0

    soft_landing_reward = proximity * speed_factor * angle_factor * contact_factor

    # ======== 3. 轻量稳定性惩罚 (stability_penalty) ========
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