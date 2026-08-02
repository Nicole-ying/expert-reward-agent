# 设计理由
**替换组件**：将 `landing_approach_reward` 整体删除，换成 `contact_success_reward`。
**为什么**：
- 审计发现 agent 把 `landing_approach_reward` exploit 了——全 truncated（20/20），len=1000，terminated=0。该奖励仅依赖高度/速度/姿态，未使用 `left_contact`/`right_contact`，agent 学会了在低空保持姿态以持续获取高额乘积奖励，而不是真正着陆。
- 缺失着陆接触信号（#5 中 `left_support_contact`/`right_support_contact` 未使用），这是根本缺口。
- 移除 exploit 源，用接触观测构建安全着陆成功奖励：仅当双腿同时接触 + 速度/姿态都在安全范围内时才给予较大正奖励（≈200），其它时间该组件输出 0。这会引导 agent 追求提前终止（触发 settled termination）而不是无限悬浮。
- 保留 `progress` 和 `soft_landing_penalty` 提供方向指引与运动平滑约束，让 agent 能逐步到达接触点附近，再通过探索获得一次性成功奖励。
**数学形式**：`contact_flag = min(left_contact, right_contact)` → 0 或 1，乘以速度/角度的高斯因子（sigma 较小，确保“安全着陆”），乘积再乘系数 200.0。未接触时奖励为 0，避免非接触时段获益。
**系数校准**：目标 per-step 主信号（progress+contact）≤ 0.3×主信号，但 success 奖励是一次性的，量级 200 能显著引导，与 episode 长度平衡合理。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Progress reward (improvement_delta on distance to origin)
    dist_curr = (x_curr**2 + y_curr**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = 10.0 * (dist_curr - dist_next)  # positive when getting closer

    # 2. Soft landing constraints
    k = 0.5
    desired_y_vel = -k * y_next
    vertical_error = y_vel_next - desired_y_vel
    penalty_y_vel = vertical_error**2
    penalty_x_vel = x_vel_next**2
    angle_error = max(0.0, abs(angle_next) - 0.2)
    penalty_angle = angle_error**2
    ang_vel_error = max(0.0, abs(ang_vel_next) - 0.5)
    penalty_ang_vel = ang_vel_error**2

    w_y_vel = 1.0
    w_x_vel = 1.0
    w_angle = 2.0
    w_ang_vel = 0.5

    soft_landing_penalty = (w_y_vel * penalty_y_vel +
                            w_x_vel * penalty_x_vel +
                            w_angle * penalty_angle +
                            w_ang_vel * penalty_ang_vel)

    # 3. Contact-based success reward (replaces landing_approach_reward)
    # Both legs must be in contact (binary) and conditions must be safe
    contact_flag = min(left_contact, right_contact)  # 0.0 or 1.0

    # Gaussian factors for safe landing: small velocity, upright, low angular velocity
    sigma_vy = 0.2
    sigma_vx = 0.2
    sigma_angle = 0.1
    sigma_angvel = 0.2

    safe_vy = 2.718281828 ** (- (y_vel_next**2) / (sigma_vy**2))
    safe_vx = 2.718281828 ** (- (x_vel_next**2) / (sigma_vx**2))
    safe_angle = 2.718281828 ** (- (angle_next**2) / (sigma_angle**2))
    safe_angvel = 2.718281828 ** (- (ang_vel_next**2) / (sigma_angvel**2))

    contact_success_reward = 200.0 * contact_flag * safe_vy * safe_vx * safe_angle * safe_angvel

    total_reward = progress - soft_landing_penalty + contact_success_reward

    components = {
        'progress': progress,
        'soft_landing_penalty': soft_landing_penalty,
        'contact_success_reward': contact_success_reward
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺失 `left_contact`/`right_contact` 信号是 agent 悬浮不落地的根因（全 truncated，零 terminated），原 `landing_approach_reward` 造成 exploit。
- **behavior**: agent 学会在低空保持姿态以持续获取高额乘积奖励，不追求双腿接触，因此 episode 永不成功终止。
- **signal**: 缺少着陆接触反馈；未使用的接触观测是填补缺口的关键。
- **level**: Level 2
- **hypothesis**: 用稀疏但高额的接触成功奖励替换 exploit 源，使 agent 只有真正安全双腿着陆才能获得最大回报，progress 和惩罚仍提供路径引导，整体应能打破悬浮循环并逐步达成任务。
- **risk**: 成功奖励稀疏可能使早期探索困难，线性 progress 可能不足以快速引导至接触点，后续可能需要增强 progress 系数或加入高度门控。