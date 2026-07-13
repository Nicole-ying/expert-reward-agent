# Response Record

`evidence`：上一轮将 `w_contact` 从 2.0 增至 8.0，当前得分 26.55（仅 target 的 13.3%），episode 全终止且 len≈350；contact_reward 均值 176.45（占比 52.4%，active_rate 仅 13.5%）与 goal_proximity 均值 -147.36（占比 43.8%，active 100%）几乎相互抵消，净奖励接近零，说明信用分配失效。
`behavior_diagnosis`：agent 在接近阶段累积大量无界负距离惩罚，到达目标后接触奖励勉强补偿，导致净信号极弱；高方差（-170~267）表明部分 episode 成功着陆、部分直接失败，学习不稳定。
`signal_completeness`：四个主职责均已存在，但 goal_proximity 作为绝对状态惩罚在接近阶段产生持续大额负奖励，使总奖励塌缩到近零区间——结构上不可达，而非职责缺失。
`selected_level`：Level 2 — 证据匹配 `state_to_improvement`：绝对距离惩罚在占据不佳状态时持续产生负值，与稀疏正奖励近乎抵消，必须将状态值变为改善量。
`selected_intervention`：唯一修改 goal_proximity，从 `-w_goal * dist_sq`（无界状态惩罚）变为 `w_goal * (dist_sq_obs - dist_sq_next)`（势能差，有界改善量），`w_goal` 同步设为 5.0 以适配新值域。
`falsifiable_hypothesis`：势能差形式使 agent 靠近目标时获正奖励、远离时获负奖励，episode 总奖励有界（~125），不再被接触奖励恰好抵消；应在接近阶段提供清晰的进展信号，使成功着陆 episode 的净奖励显著为正，失败 episode 的净奖励接近零而非深负。
`expected_next_round`：goal_proximity 的 `episode_sum_mean` 应变为正值且 bounded，整体得分上升、方差下降，contact_reward 的 `signed_share` 不再被迫与 goal_proximity 符号对抗。
`main_risk`：每步势能差量级可能过小（~0.4/步），早期接近梯度不足以对抗其他惩罚导致徘徊；若出现此情况，下一轮可上调 `w_goal` 或为 contact_reward 增设速度/角度约束防止冲击刷分。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation components
    x = obs[0]            # horizontal position relative to target pad
    y = obs[1]            # vertical position relative to target pad
    vx = obs[2]           # horizontal velocity
    vy = obs[3]           # vertical velocity
    angle = obs[4]        # body angle
    ang_vel = obs[5]      # angular velocity
    left_contact = obs[6] # left support leg contact (0 or 1)
    right_contact = obs[7]# right support leg contact (0 or 1)

    # next_obs
    nx = next_obs[0]
    ny = next_obs[1]

    # Hyperparameters
    w_goal = 5.0                    # potential-based progress weight
    alpha_proximity = 5.0           # controls the activation radius for proximity-based terms
    w_vel = 0.5
    w_angle = 0.2
    w_angvel = 0.1
    w_contact = 8.0

    # Squared distances for potential computation
    dist_sq_obs = x**2 + y**2
    dist_sq_next = nx**2 + ny**2

    # Soft proximity weight based on current position
    proximity = 1.0 / (1.0 + alpha_proximity * dist_sq_obs)

    # 1. Main progress: potential-based improvement toward target center
    #    Positive when moving closer, negative when moving away, bounded total
    goal_proximity = w_goal * (dist_sq_obs - dist_sq_next)

    # 2. Soft landing velocity penalty: active only near the target
    velocity_penalty = -w_vel * (vx**2 + vy**2) * proximity

    # 3. Orientation stability penalty: penalize tilt and spin everywhere (light weight)
    orientation_penalty = -w_angle * (angle**2) - w_angvel * (ang_vel**2)

    # 4. Contact reward: encourage both legs touching, gated by proximity
    both_legs_contact = left_contact * right_contact  # 1 only if both are 1
    contact_reward = w_contact * both_legs_contact * proximity

    # Total reward
    total_reward = goal_proximity + velocity_penalty + orientation_penalty + contact_reward

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_reward": contact_reward
    }

    return float(total_reward), components
```
