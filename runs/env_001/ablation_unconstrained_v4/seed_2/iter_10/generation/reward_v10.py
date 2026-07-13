def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x_new = next_obs[0]
    y_new = next_obs[1]
    vx_new = next_obs[2]
    vy_new = next_obs[3]
    angle_new = next_obs[4]
    angvel_new = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 高度反比奖励（替代 height_near，急剧引导最终下降）----------
    height_reward = 8.0 / (1.0 + y_new ** 2)

    # ---------- 低空强化对准（高度越低，对准越重要）----------
    alignment_gate = 1.0 / (1.0 + max(0.0, y_new))
    alignment = 2.5 * alignment_gate / (1.0 + 12.0 * (x_new ** 2))

    # ---------- 下降进度奖励 ----------
    delta_y = obs[1] - y_new
    progress_y = 0.5 * max(0.0, delta_y)

    # ---------- 接触奖励（单腿即奖，双腿加磅）----------
    any_contact = max(left_contact, right_contact)
    both_contact = left_contact * right_contact
    contact_reward = 2.0 * any_contact + 12.0 * both_contact

    # ---------- 低空垂直速度惩罚（软着陆约束）----------
    low_alt = max(0.0, 1.5 - y_new)
    soft_landing_penalty = -8.0 * low_alt * (max(0.0, -vy_new) ** 2)

    # ---------- 水平速度惩罚 ----------
    lat_penalty = -8.0 * (max(0.0, abs(vx_new) - 0.25) ** 2)

    # ---------- 姿态角惩罚 ----------
    att_penalty = -6.0 * (abs(angle_new) ** 2)

    # ---------- 角速度惩罚 ----------
    angvel_penalty = -3.0 * (abs(angvel_new) ** 2)

    total_reward = (
        height_reward +
        alignment +
        progress_y +
        contact_reward +
        soft_landing_penalty +
        lat_penalty +
        att_penalty +
        angvel_penalty
    )

    components = {
        'height_reward': height_reward,
        'alignment': alignment,
        'progress_y': progress_y,
        'contact_reward': contact_reward,
        'soft_landing_penalty': soft_landing_penalty,
        'lat_penalty': lat_penalty,
        'att_penalty': att_penalty,
        'angvel_penalty': angvel_penalty
    }

    return float(total_reward), components