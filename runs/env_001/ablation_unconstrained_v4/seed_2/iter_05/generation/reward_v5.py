def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 读取状态 ----------
    x_old, y_old = obs[0], obs[1]
    x_new = next_obs[0]
    y_new = next_obs[1]
    vx_new = next_obs[2]
    vy_new = next_obs[3]
    angle_new = next_obs[4]
    angvel_new = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 辅助量 ----------
    dist_old = (x_old**2 + y_old**2)**0.5
    dist_new = (x_new**2 + y_new**2)**0.5
    any_contact = max(left_contact, right_contact)
    double_contact = left_contact * right_contact

    # ---------- 进度奖励（鼓励持续下降和靠近）----------
    w_progress = 5.0
    w_y_progress = 10.0
    progress = w_progress * (dist_old - dist_new)
    y_progress = w_y_progress * (y_old - y_new)

    # ---------- 高度门控的水平对准（关键修改）----------
    # 只有当 y_new < height_threshold 时，alignment 才会激活，
    # 且奖励随高度降低线性增大，迫使 agent 先下降再对准
    height_threshold = 0.8
    height_gate = max(0.0, 1.0 - y_new / height_threshold)  # y_new >= 0.8 时=0，y_new=0 时=1
    alignment_raw = 1.0 / (1.0 + x_new**2)
    w_alignment = 0.8
    alignment = w_alignment * alignment_raw * height_gate

    # ---------- 接近地面鼓励（无接触门槛）----------
    w_height_near = 4.0
    ground_threshold = 0.8   # 使用同一阈值，使奖励在下降至0.8以下时持续增加
    height_near_reward = w_height_near * max(0.0, ground_threshold - y_new)

    # ---------- 接触奖励 ----------
    w_any_contact = 5.0
    any_contact_reward = w_any_contact * any_contact

    # ---------- 双腿稳定着陆奖励 ----------
    w_stable = 20.0
    vx_tol, vy_tol, angle_tol, angvel_tol = 0.2, 0.2, 0.1, 0.1
    stability = (max(0.0, 1.0 - abs(vx_new)/vx_tol) *
                 max(0.0, 1.0 - abs(vy_new)/vy_tol) *
                 max(0.0, 1.0 - abs(angle_new)/angle_tol) *
                 max(0.0, 1.0 - abs(angvel_new)/angvel_tol))
    stable_contact_reward = w_stable * double_contact * stability

    # ---------- 水平速度惩罚（辅助对准）----------
    w_horiz_penalty = 0.2
    horizontal_penalty = -w_horiz_penalty * (vx_new**2)

    # ---------- 软安全约束（速度/姿态越界）----------
    w_vel = 0.05
    w_att = 0.05
    vx_max, vy_max = 0.5, 0.5
    angle_max, angvel_max = 0.2, 0.2

    excess_vx = max(0.0, abs(vx_new) - vx_max)
    excess_vy = max(0.0, abs(vy_new) - vy_max)
    vel_penalty = -w_vel * (excess_vx**2 + excess_vy**2)

    excess_angle = max(0.0, abs(angle_new) - angle_max)
    excess_angvel = max(0.0, abs(angvel_new) - angvel_max)
    att_penalty = -w_att * (excess_angle**2 + excess_angvel**2)

    # ---------- 总奖励 ----------
    total_reward = (progress + y_progress + alignment +
                    height_near_reward + any_contact_reward +
                    stable_contact_reward +
                    horizontal_penalty + vel_penalty + att_penalty)

    components = {
        'progress': progress,
        'y_progress': y_progress,
        'alignment': alignment,
        'height_near_reward': height_near_reward,
        'any_contact_reward': any_contact_reward,
        'stable_contact_reward': stable_contact_reward,
        'horizontal_penalty': horizontal_penalty,
        'vel_penalty': vel_penalty,
        'att_penalty': att_penalty
    }
    return float(total_reward), components