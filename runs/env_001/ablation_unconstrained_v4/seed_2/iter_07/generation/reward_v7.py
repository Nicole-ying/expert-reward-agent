def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 读取状态 ----------
    x_old, y_old = obs[0], obs[1]
    x_new, y_new = next_obs[0], next_obs[1]
    vx_new = next_obs[2]
    vy_new = next_obs[3]
    angle_new = next_obs[4]
    angvel_new = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 全局前进引导（向原点缩短距离）----------
    dist_old = (x_old**2 + y_old**2)**0.5
    dist_new = (x_new**2 + y_new**2)**0.5
    progress = 3.0 * (dist_old - dist_new)

    # ---------- 高度近地奖励（平方形态，宽范围）----------
    height_threshold = 1.5
    height_near_raw = max(0.0, height_threshold - y_new)
    w_height = 5.0
    height_near = w_height * (height_near_raw ** 2)

    # ---------- 水平对准（依赖高度的 gate）----------
    height_gate = max(0.0, 1.0 - y_new / height_threshold)
    alignment_raw = 1.0 / (1.0 + x_new**2)
    w_align = 0.5
    alignment = w_align * alignment_raw * height_gate

    # ---------- 接触奖励 ----------
    any_contact = max(left_contact, right_contact)
    w_contact = 2.0
    contact_reward = w_contact * any_contact

    # ---------- 水平速度惩罚（hinge）----------
    vx_limit = 0.2
    lat_penalty = -2.0 * (max(0.0, abs(vx_new) - vx_limit) ** 2)

    # ---------- 下坠速度惩罚（hinge，防止过快下降）----------
    vy_down_limit = 0.5
    down_penalty = -2.0 * (max(0.0, -vy_new - vy_down_limit) ** 2)

    # ---------- 姿态与角速度惩罚（hinge）----------
    angle_limit = 0.1
    att_penalty = -2.0 * (max(0.0, abs(angle_new) - angle_limit) ** 2)
    angvel_limit = 0.1
    angvel_penalty = -2.0 * (max(0.0, abs(angvel_new) - angvel_limit) ** 2)

    # ---------- 总奖励 ----------
    total_reward = (progress + height_near + alignment + contact_reward +
                    lat_penalty + down_penalty + att_penalty + angvel_penalty)

    components = {
        'progress': progress,
        'height_near': height_near,
        'alignment': alignment,
        'contact_reward': contact_reward,
        'lat_penalty': lat_penalty,
        'down_penalty': down_penalty,
        'att_penalty': att_penalty,
        'angvel_penalty': angvel_penalty
    }
    return float(total_reward), components