# 设计理由
本轮的信号覆盖审计发现，`body_angle`（obs[4]）未被使用，但当前策略已稳定着陆（terminated 20/20，len=289），没有明显的姿态或接触问题，因此不需要新增组件。所有六个预判均为✅，说明当前方向正确，只需在现有骨架上微调信号强度。

**证据**：`landing_incentive` 是主导奖励（96.3%），但其只在腿接触地面时触发（active_rate 15.5%），导致大部分 step 仅依靠极其微弱的 `progress_reward`（per‑step 约 0.005）。若能适度放大 `progress_reward`，可以在下降阶段提供更强的方向激励，进一步缩短 episode 长度并提高总分，同时不会侵占着陆阶段的奖励，因为 `landing_incentive` 在着地瞬间的峰值远高于一路上的 `progress` 积累量。

**修改内容**：将 `w_progress` 从 1.0 提高到 5.0。  
按当前统计，主信号 `landing_incentive` 的 per‑step 均值约为 0.126；新 `progress_reward` 的 per‑step 均值约为 \( 0.00477 \times 5 = 0.024 \)，仅为主信号的 19%，远低于 0.3 倍的安全阈值，不会支配学习。同时保留原有的 `angvel_penalty`（无需调整，因其触发率极低但无害）。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next observation
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_vx = next_obs[2]
    next_vy = next_obs[3]
    next_angle = next_obs[4]
    next_angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- Progress: distance reduction (coefficient increased) ---
    dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    w_progress = 5.0
    progress = dist - next_dist

    # --- Landing incentive: only when legs touch ground ---
    leg_contact = 1.0 if (left_contact > 0.5 or right_contact > 0.5) else 0.0
    speed = (next_vx ** 2 + next_vy ** 2) ** 0.5
    speed_factor = 1.0 / (1.0 + 3.0 * speed)
    w_landing = 1.0
    landing_incentive = leg_contact * w_landing / (1.0 + next_dist * 5.0) * speed_factor

    # --- Angular velocity penalty (unchanged) ---
    w_angvel = 0.05
    safe_angvel = 0.5
    angvel_error = abs(next_angvel) - safe_angvel
    angvel_penalty = -w_angvel * angvel_error if angvel_error > 0 else 0.0

    # --- Total reward ---
    total_reward = w_progress * progress + landing_incentive + angvel_penalty

    components = {
        "progress_reward": w_progress * progress,
        "landing_incentive": landing_incentive,
        "angvel_penalty": angvel_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 未使用的 `body_angle` 对当前终止模式无解释力；所有关键观测均已覆盖，信号齐全。
- **behavior**: agent 已掌握快速着陆策略，episode 以成功 termination 结束，但下降阶段缺少足够的即时进度激励。
- **signal**: `progress_reward` 过于微弱（per‑step 0.005 vs 主信号 0.126），导致下降动力不足，episode 长度仍有压缩空间。
- **level**: Level 1（尺度修复——提高进度系数）
- **hypothesis**: 将进度系数放大至 5.0 能在下降全程提供更清晰的梯度，缩短 episode 并小幅推高总分，而着陆瞬间的强奖励仍可保证稳定着陆质量。
- **risk**: 若 agent 过度追求进度积累，可能在接近地面时仍保持较大下降速度，略微增加着陆冲击；但 `landing_incentive` 的接触门控和速度抑制因子会继续约束其着陆姿态，风险轻微。