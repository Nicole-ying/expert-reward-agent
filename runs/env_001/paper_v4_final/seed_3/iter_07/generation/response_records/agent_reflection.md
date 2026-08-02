# Response Record

# 设计理由
**只修改了 `lateral_pos_penalty` 的系数**，从 0.6 降到 0.08。  
- 证据：本轮训练反馈中，`lateral_pos_penalty` 的 episode_sum_mean 高达 -57.1，per‑step ≈ -0.061，而主信号 `progress_gated` 的 per‑step ≈ +0.030。惩罚强度是正向奖励的 2 倍以上，直接导致总分变负（-13.38）。  
- 根因：在第 6 轮将 `w_lat_pos` 从约 0.15（推测）大幅提升到 0.6，预期改善中心对准，但实际过强的惩罚压垮了整个奖励结构，使 agent 的步长从 668 延长到 936，得分从 +30.06 跌至 -13.38。  
- 修改：将系数降为 0.08，使 per‑step 惩罚 ≈ -0.008，与主信号比值约 0.27，符合“惩罚 ≤ 主信号 0.3x”的校准规则。该值接近第 5 轮（best 30.06）时的横向惩罚强度，有望恢复正向激励平衡。  
- 未改动其它组件，符合“正常模式每次改一个组件”的约束。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # observation indices: 0:x, 1:y, 2:vx, 3:vy, 4:angle, 5:ang_vel, 6:left_contact, 7:right_contact
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_vel = next_obs[5]
    n_lc = next_obs[6]  # left contact
    n_rc = next_obs[7]  # right contact

    # ---------- 1. progress towards origin (distance decrease) ----------
    dist_old = (x**2 + y**2) ** 0.5
    dist_new = (nx**2 + ny**2) ** 0.5
    progress = dist_old - dist_new
    w_progress = 30.0

    # ---------- 2. attitude gate: suppress progress when angle is dangerous ----------
    abs_angle = abs(n_angle)
    angle_gate = 1.0 - 0.8 * (2.718281828 ** (12.0 * (abs_angle - 0.30)) / 
                              (1.0 + 2.718281828 ** (12.0 * (abs_angle - 0.30))))

    # ---------- 3. lateral position penalty (encourage centering) ----------
    w_lat_pos = 0.08          # ← 唯一修改：从 0.6 降为 0.08
    lateral_pos_penalty = -w_lat_pos * (nx ** 2)

    # ---------- 4. angular velocity penalty (smooth rotation) ----------
    w_angvel = 0.05
    angvel_penalty = -w_angvel * (n_ang_vel ** 2)

    # ---------- 5. contact-based landing proxy ----------
    mean_contact = (n_lc + n_rc) / 2.0

    k_y = 10.0
    k_vy = 8.0
    k_ang = 15.0
    f_y   = 1.0 / (1.0 + k_y   * abs(ny))
    f_vy  = 1.0 / (1.0 + k_vy  * abs(nvy))
    f_ang = 1.0 / (1.0 + k_ang * abs(n_angle))

    contact_landing_factor = (mean_contact * f_y * f_vy * f_ang) ** 0.25
    w_contact_land = 5.0
    contact_landing_reward = w_contact_land * contact_landing_factor

    # ---------- combine ----------
    total_reward = (w_progress * progress * angle_gate
                    + lateral_pos_penalty
                    + angvel_penalty
                    + contact_landing_reward)

    components = {
        "progress_gated": w_progress * progress * angle_gate,
        "lateral_pos_penalty": lateral_pos_penalty,
        "angvel_penalty": angvel_penalty,
        "contact_landing_reward": contact_landing_reward
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 所有观测已被使用，无信号缺口；绝大部分 episode 因超时截断，agent 存活但未能着陆。`lateral_pos_penalty` 是唯一压倒性负贡献组件（占 magnitude share 47.5%，per‑step 为主信号的 2×），导致总分塌陷。  
- **behavior**: agent 长时间徘徊（len≈937），其进步奖励（+27.7）和着陆奖励（+21.2，但 active_rate 仅 0.8%）全部被过强的横向二次惩罚（-57.1）吞噬。  
- **signal**: 横向位置惩罚系数过大，严重压制正向信号。  
- **level**: Level 1 — 尺度修复（降低惩罚系数）。  
- **hypothesis**: 将 `w_lat_pos` 回调至接近第 5 轮水平，惩罚强度降至主信号的 0.3x 以内，总奖励将恢复正向驱动，agent 不再为避免惩罚而牺牲进步效率，从而缩短 episode 长度、提高得分。  
- **risk**: 若横向引导过弱，agent 可能在着陆后期出现较大 x 偏移，但温和惩罚仍足以维持大致居中，未来可配合 hinge 门控进一步优化。
