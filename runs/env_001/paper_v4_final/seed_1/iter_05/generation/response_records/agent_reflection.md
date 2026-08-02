# Response Record

# 设计理由
当前 agent 在所有 episode 中 landing_incentive 的 active_rate 为 100%，说明 agent 几乎每一步都拿到了这个奖励，包括未接触地面的悬停阶段（因为 contact_gate 存在 0.1 的底值）。这导致 agent 学会了长时间在目标点附近徘徊，部分 episode 被时间截断而未能真正触发成功着陆终止（body_not_awake_or_settled）。因此 episode 长度偏高（847.7），虽然得分已达到 180.66，但距离目标 200 仍有差距，本质上是任务完成率不够高。

本次修改（Level 2 结构变换）：
- **组件**：`landing_incentive` 中的接触门控。
- **变换**：将 `contact_gate = 0.1 + 0.9 * leg_contact` 改为纯二进制 `float(leg_contact > 0.5)`（1 或 0）。即没有支撑腿接触时完全取消该奖励，仅保留 `progress_reward` 作为距离导向。同时将 `w_landing` 从 0.5 提升至 1.0，以补偿步数可能缩短造成的累积奖励下降，并更强烈地激励有接触的低速状态。
- **数学形式**：`landing_incentive = leg_contact * 1.0 / (1 + next_dist * 5.0) * speed_factor`
- **系数校准**：接触后单步最大奖励约 1.0（dist≈0, speed≈0），约为主信号当前 per-step（~0.29）的 3.4 倍，仍在可控范围；无接触时该组件归零，总奖励仅剩 `progress_reward`，不会完全丧失梯度。角度惩罚负担很低，不违反设计约束。
- **风险与缓解**：探索初期可能因无接触时奖励极低而学习变慢，但 `progress_reward`（距离缩减）仍能提供朝向目标垫的梯度，agent 很容易发现触地行为；同时 `speed_factor` 保证只有低速接触才能拿到高分，防止猛烈坠落 exploit。

# 代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next observation
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_angle = next_obs[4]
    next_vx = next_obs[2]
    next_vy = next_obs[3]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- Progress: distance reduction ---
    dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    w_progress = 1.0
    progress = dist - next_dist

    # --- Landing incentive: only when legs touch ground ---
    leg_contact = 1.0 if (left_contact > 0.5 or right_contact > 0.5) else 0.0
    # Speed magnitude (linear velocities)
    speed = (next_vx ** 2 + next_vy ** 2) ** 0.5
    speed_factor = 1.0 / (1.0 + 3.0 * speed)
    w_landing = 1.0
    landing_incentive = leg_contact * w_landing / (1.0 + next_dist * 5.0) * speed_factor

    # --- Health constraint: body angle ---
    w_angle = 0.5
    safe_angle = 0.3
    angle_error = abs(next_angle) - safe_angle
    angle_penalty = -w_angle * angle_error if angle_error > 0 else 0.0

    # --- Total reward ---
    total_reward = w_progress * progress + landing_incentive + angle_penalty

    components = {
        "progress_reward": w_progress * progress,
        "landing_incentive": landing_incentive,
        "angle_penalty": angle_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 所有观测维度均已使用，信号齐全；`landing_incentive` 的 0.1 底值使 agent 在无接触悬停时仍能获取可观奖励，导致延迟着陆和截断。
- **behavior**: agent 学会向目标垫移动并减速，但通常在空中徘徊，部分 episode 被时间截断而未能触发成功着陆终止。
- **signal**: `landing_incentive` 的接触门控太宽松，缺少对“必须实际触地”的强制要求。
- **level**: Level 2
- **hypothesis**: 移除无接触奖励后，agent 必须降落才能获得主奖励，从而加快成功着陆、缩短 episode 并提高 task completion 比例，总分应逼近 200。
- **risk**: 探索阶段可能因无接触时总奖励极低而短期性能波动，但 `progress_reward` 仍能提供下降方向的梯度，风险可控。
