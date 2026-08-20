def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：基于位置进步的 potential-based shaping
    dist_old = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_new = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_reward = dist_old - dist_new

    # 着陆引导与接触奖励
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    left_contact_old = obs[6]
    right_contact_old = obs[7]
    left_contact_new = next_obs[6]
    right_contact_new = next_obs[7]

    # 进入着陆区域（靠近中心）才施加精细控制
    dist = (x ** 2 + y ** 2) ** 0.5
    near_pad = float(dist < 2.0)

    # 速度惩罚：鼓励静止
    speed = (vx ** 2 + vy ** 2) ** 0.5
    velocity_penalty = -0.1 * speed * near_pad

    # 角度惩罚：鼓励竖直
    angle_abs = abs(angle)
    angle_penalty = -0.1 * angle_abs * near_pad

    # 接触变化奖励（一次性，鼓励支撑脚接触）
    contact_change = max(0.0, left_contact_new - left_contact_old) + max(0.0, right_contact_new - right_contact_old)
    contact_bonus = 10.0 * contact_change

    stable_landing_reward = velocity_penalty + angle_penalty + contact_bonus

    total_reward = progress_reward + stable_landing_reward

    components = {
        "progress_reward": progress_reward,
        "stable_landing_reward": stable_landing_reward
    }

    return float(total_reward), components