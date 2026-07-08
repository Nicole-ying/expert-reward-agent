def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：每一步向目标靠近的 progress（potential-based shaping）
    dist_old = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_new = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_reward = dist_old - dist_new

    # 稳定着陆 proxy：多条件连续乘积，引导低速、竖直、双脚接触且对准着陆垫中心
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 温度参数（根据环境大致范围设定，可在后续迭代中调整）
    sigma_p = 1.0    # 位置衰减尺度
    sigma_v = 1.0    # 速度衰减尺度
    sigma_a = 0.3    # 角度衰减尺度（弧度）

    # 距离、速度、角度项的指数和
    exponent = -(
        (x ** 2 + y ** 2) / (2.0 * sigma_p ** 2) +
        (vx ** 2 + vy ** 2) / (2.0 * sigma_v ** 2) +
        (angle ** 2) / (2.0 * sigma_a ** 2)
    )

    contact_factor = (left_contact + right_contact) / 2.0
    # 使用 2.718281828**exponent 替代 exp，避免 numpy 依赖
    stable_landing_reward = 10.0 * contact_factor * (2.718281828 ** exponent)

    total_reward = progress_reward + stable_landing_reward

    components = {
        "progress_reward": progress_reward,
        "stable_landing_reward": stable_landing_reward
    }

    return float(total_reward), components