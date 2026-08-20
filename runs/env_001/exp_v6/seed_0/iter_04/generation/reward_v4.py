def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 目标位置 (0,0)，obs 中 x,y 是相对于目标平台的坐标
    target_pos = (0.0, 0.0)

    # 1. 主学习信号：进度奖励（保持上轮的 scale=10，不做改动）
    dist = ((obs[0] - target_pos[0]) ** 2 + (obs[1] - target_pos[1]) ** 2) ** 0.5
    next_dist = ((next_obs[0] - target_pos[0]) ** 2 + (next_obs[1] - target_pos[1]) ** 2) ** 0.5
    progress_reward = (dist - next_dist) * 10.0

    # 2. 稳定性惩罚（保持不变）
    vel_x = abs(next_obs[2])
    vel_y = abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    stability_penalty = -0.001 * (vel_x + vel_y) - 0.001 * angle - 0.001 * ang_vel

    # 3. 软着陆 proxy：从"每步状态奖励"改为"一次性触地事件奖励"
    #    诊断：上轮 nonzero_rate=44%，agent 着陆后静坐收割 0.5/步，ratio=4.81 主导总奖励。
    #    修复：通过比较 obs vs next_obs 的腿接触状态，检测「着陆瞬间」——
    #    双足从未接触变为接触的那一步才给奖励，消除静坐 exploit。
    prev_both_contact = (obs[6] > 0.5 and obs[7] > 0.5)
    curr_both_contact = (next_obs[6] > 0.5 and next_obs[7] > 0.5)
    just_landed = curr_both_contact and not prev_both_contact

    # 着陆质量条件（阈值适当放宽，因为是事件触发，不用担心每步 exploit）
    near_target = (next_dist < 0.3)
    low_speed = (abs(next_obs[2]) + abs(next_obs[3]) < 0.5)
    stable_angle = (abs(next_obs[4]) < 0.2)

    soft_landing_proxy = 1.0 if (just_landed and near_target and low_speed and stable_angle) else 0.0

    total_reward = progress_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }
    return float(total_reward), components