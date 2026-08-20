def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    cx, cy = float(obs[0]), float(obs[1])
    nx, ny = float(next_obs[0]), float(next_obs[1])

    d_curr = (cx ** 2 + cy ** 2) ** 0.5
    d_next = (nx ** 2 + ny ** 2) ** 0.5

    # 1. 主学习信号：回退到 best 的 scale=8.0
    #    iter 9 把 scale 提到 15.0 但得分微降，说明瓶颈不在 progress 强度
    progress_reward = (d_curr - d_next) * 8.0

    # 2. 软着陆近似信号：数学形式从乘积改为 min() * contact
    #    乘积形式的问题：当三个因子中有一个很差时，整个 proxy≈0，
    #    且梯度被其他因子衰减。min() 让梯度直接流向瓶颈维度。
    vx, vy = float(next_obs[2]), float(next_obs[3])
    speed = (vx ** 2 + vy ** 2) ** 0.5
    angle = abs(float(next_obs[4]))

    proximity_factor = max(0.0, 1.0 - d_next / 0.3)
    speed_factor = max(0.0, 1.0 - speed / 0.5)
    angle_factor = max(0.0, 1.0 - angle / 0.3)
    contact_factor = (
        max(0.0, float(next_obs[6])) * max(0.0, float(next_obs[7]))
    )

    # 用 min 替代乘积：agent 必须同时满足三个条件才能获得信号，
    # 但梯度始终流向当前最差的那个维度（瓶颈），不被其他因子衰减
    approach_quality = min(proximity_factor, speed_factor, angle_factor)
    soft_landing_proxy = approach_quality * contact_factor

    # 3. 中心邻近奖励（保持 best 配置不变）
    center_bonus = 0.2 * max(0.0, 1.0 - d_next / 0.5)

    total_reward = progress_reward + soft_landing_proxy + center_bonus

    components = {
        "progress_reward": progress_reward,
        "soft_landing_proxy": soft_landing_proxy,
        "center_bonus": center_bonus,
    }
    return float(total_reward), components