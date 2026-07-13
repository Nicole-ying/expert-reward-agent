def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 位置和距离
    x_obs, y_obs = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    dist_obs = (x_obs**2 + y_obs**2)**0.5
    dist_next = (x_next**2 + y_next**2)**0.5

    # 接触状态（基于 next_obs，反映动作后的着陆情况）
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    both_landed = (left_contact > 0.5) and (right_contact > 0.5)

    # 姿态角度
    angle_next = next_obs[4]

    # ========== 组件定义 ==========

    # 1. 双足着陆奖励：每步给予常数奖励，鼓励保持接触直到终止
    w_contact = 1.0
    contact_reward = w_contact * (1.0 if both_landed else 0.0)

    # 2. 目标接近状态奖励（凸化，密集梯度）
    goal_threshold = 2.0
    proximity = max(0.0, 1.0 - dist_next / goal_threshold)
    w_proximity = 0.5
    proximity_reward = w_proximity * proximity

    # 3. 差分进度奖励（保持原有正向差分思想）
    progress = max(0.0, dist_obs - dist_next)
    w_progress = 1.5
    progress_reward = w_progress * progress

    # 4. 引擎使用惩罚（节约燃料）
    w_engine = 0.03
    engine_penalty = -w_engine if action != 0 else 0.0

    # 5. 姿态角度惩罚（hinge，允许小幅摇摆）
    allowed_angle = 0.2  # 约11.5度
    excess_angle = max(0.0, abs(angle_next) - allowed_angle)
    w_angle = 0.1
    upright_penalty = -w_angle * (excess_angle ** 2)

    # 总奖励
    total_reward = contact_reward + proximity_reward + progress_reward + engine_penalty + upright_penalty

    reward_components = {
        'contact_reward': contact_reward,
        'proximity_reward': proximity_reward,
        'progress_reward': progress_reward,
        'engine_penalty': engine_penalty,
        'upright_penalty': upright_penalty
    }

    return float(total_reward), reward_components