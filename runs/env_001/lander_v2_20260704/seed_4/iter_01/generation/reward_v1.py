def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 计算当前位置与目标(0,0)的欧氏距离
    dist_obs = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    # 主学习信号：每一步靠近目标的进展
    progress_reward = dist_obs - dist_next

    # 稳定/安全约束：抑制过度速度、倾角和角速度
    w_v = 0.01   # 线速度惩罚权重
    w_a = 0.01   # 倾角惩罚权重
    w_w = 0.01   # 角速度惩罚权重
    stability_penalty = (
        -w_v * (abs(next_obs[2]) + abs(next_obs[3]))
        - w_a * abs(next_obs[4])
        - w_w * abs(next_obs[5])
    )

    # 任务完成近似信号：多条件组合的着陆奖励代理
    x_thresh = 0.1
    y_thresh = 0.1
    v_thresh = 0.2
    angle_thresh = 0.1
    left_contact = next_obs[6] > 0.5
    right_contact = next_obs[7] > 0.5
    if (abs(next_obs[0]) < x_thresh and abs(next_obs[1]) < y_thresh and
        abs(next_obs[2]) < v_thresh and abs(next_obs[3]) < v_thresh and
        abs(next_obs[4]) < angle_thresh and left_contact and right_contact):
        landing_bonus = 0.1
    else:
        landing_bonus = 0.0

    total_reward = progress_reward + stability_penalty + landing_bonus

    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'landing_bonus': landing_bonus
    }

    return float(total_reward), components