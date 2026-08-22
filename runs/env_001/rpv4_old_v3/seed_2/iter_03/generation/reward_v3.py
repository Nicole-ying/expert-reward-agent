def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 0. 解析观测
    x_pos, y_pos = obs[0], obs[1]
    body_angle = obs[4]

    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]

    # 1. 距离势能塑造 (progress_shaping)
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    potential_curr = -dist
    potential_next = -next_dist
    progress_reward = 5.0 * (potential_next - potential_curr)

    # 2. 成功着陆奖励 (success_landing_bonus)
    contact_any = (next_left_contact > 0.5) or (next_right_contact > 0.5)
    landed = (
        next_dist < 0.1 and
        abs(next_x_vel) < 0.2 and
        abs(next_y_vel) < 0.2 and
        abs(next_body_angle) < 0.1 and
        contact_any
    )
    success_bonus = 100.0 if landed else 0.0

    # 3. 坠毁/越界惩罚 (crash_penalty)
    out_of_bounds = abs(next_x_pos) > 1.5
    # 接触地面但未满足软着陆条件，或出界，均视为失败
    crash = (not landed) and (contact_any or out_of_bounds)
    crash_penalty_val = -100.0 if crash else 0.0

    # 4. 姿态惩罚
    angle_penalty_val = -0.5 * (body_angle ** 2)

    # 5. 动作代价
    action_cost_val = -0.02 if action != 0 else 0.0

    # 总奖励
    total_reward = (
        progress_reward +
        success_bonus +
        crash_penalty_val +
        angle_penalty_val +
        action_cost_val
    )

    components = {
        "progress_shaping": progress_reward,
        "success_landing_bonus": success_bonus,
        "crash_penalty": crash_penalty_val,
        "angle_penalty": angle_penalty_val,
        "action_cost": action_cost_val
    }

    return float(total_reward), components