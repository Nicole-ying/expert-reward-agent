def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 读取状态
    x_old, y_old = obs[0], obs[1]
    x_new, y_new = next_obs[0], next_obs[1]
    vx_new = next_obs[2]
    vy_new = next_obs[3]
    angle_new = next_obs[4]
    angvel_new = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 高度近地奖励（受着陆质量 gate 调制）----------
    height_threshold = 2.0
    height_near_raw = max(0.0, height_threshold - y_new)
    # 着陆质量因子：双腿平均接触度 × 姿态竖直度
    contact_factor = (left_contact + right_contact) / 2.0
    angle_factor = 1.0 / (1.0 + 10.0 * abs(angle_new))
    landing_quality = contact_factor * angle_factor
    height_near = 0.5 * (height_near_raw ** 2) * landing_quality

    # ---------- 水平对准（全程激励，不依赖高度）----------
    alignment = 2.0 / (1.0 + 10.0 * (x_new ** 2))

    # ---------- 接触奖励（双腿同时接触时给予高奖励）----------
    both_contact = left_contact * right_contact
    contact_reward = 5.0 * both_contact

    # ---------- 垂直速度惩罚（更严格的下降限速）----------
    vy_limit = 0.3
    down_penalty = -5.0 * (max(0.0, -vy_new - vy_limit) ** 2)

    # ---------- 水平速度惩罚 ----------
    vx_limit = 0.2
    lat_penalty = -2.0 * (max(0.0, abs(vx_new) - vx_limit) ** 2)

    # ---------- 姿态角惩罚 ----------
    angle_limit = 0.05
    att_penalty = -2.0 * (max(0.0, abs(angle_new) - angle_limit) ** 2)

    # ---------- 角速度惩罚 ----------
    angvel_limit = 0.05
    angvel_penalty = -2.0 * (max(0.0, abs(angvel_new) - angvel_limit) ** 2)

    # ---------- 趋近静止奖励（鼓励最终稳定）----------
    speed_norm = (vx_new ** 2 + vy_new ** 2) ** 0.5
    still_bonus = 0.5 * max(0.0, 0.1 - speed_norm)

    total_reward = (
        height_near +
        alignment +
        contact_reward +
        down_penalty +
        lat_penalty +
        att_penalty +
        angvel_penalty +
        still_bonus
    )

    components = {
        'height_near': height_near,
        'alignment': alignment,
        'contact_reward': contact_reward,
        'down_penalty': down_penalty,
        'lat_penalty': lat_penalty,
        'att_penalty': att_penalty,
        'angvel_penalty': angvel_penalty,
        'still_bonus': still_bonus
    }

    return float(total_reward), components