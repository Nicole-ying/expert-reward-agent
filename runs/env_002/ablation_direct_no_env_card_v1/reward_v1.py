def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── 主学习信号：向前速度 ──
    forward_velocity = next_obs[2]
    w_forward = 1.0
    forward_reward = w_forward * forward_velocity

    # ── 姿态约束：保持身体直立 (hinge 允许小幅摆动) ──
    hull_angle = next_obs[0]
    w_angle = 0.5
    angle_threshold = 0.2   # 允许 ±0.2 rad 的自由摆动区间
    posture_penalty = -w_angle * max(0.0, abs(hull_angle) - angle_threshold)

    # ── 能耗代价：动作平方和 ──
    w_energy = 0.01
    action_penalty = -w_energy * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)

    # 总奖励
    total_reward = forward_reward + posture_penalty + action_penalty

    components = {
        "forward_reward": forward_reward,
        "posture_penalty": posture_penalty,
        "action_penalty": action_penalty
    }
    return float(total_reward), components