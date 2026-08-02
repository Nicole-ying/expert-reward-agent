# 设计理由
当前奖励函数中 **contact_landing_reward 占据了 89.9% 的奖励贡献**，而 progress 信号（progress_gated）在 episode 中总计仅 7.7，导致 agent 几乎完全被最后的接触奖励驱动，在前期路径规划上缺乏效率。同时 lateral_pos_penalty 只占总惩罚的 5.8%，不足以迫使 agent 飞向中心。由于环境得分很可能重视“快速、平稳且准确着陆”，我们需要**提升 progress 和 centering 惩罚的占比，让 agent 在飞行过程中更积极地朝向原点并保持中心对准，而不是仅追求末端接触**。本次采用 Level 1 尺度修复：将 `w_progress` 从 10 提高到 30，`w_lat_pos` 从 0.15 提高到 0.6，其余组件保持不变。修改后 progress 的 episode 预期贡献从约 7.7 升至 23.1，lateral 惩罚从约 -18 升至 -72，但仍远小于 contact_landing_reward 的主信号（≈280），不会淹没着陆激励，同时可为快速、居中的路径提供足够梯度。

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
    w_progress = 30.0  # scaled up from 10.0

    # ---------- 2. attitude gate: suppress progress when angle is dangerous ----------
    abs_angle = abs(n_angle)
    angle_gate = 1.0 - 0.8 * (2.718281828 ** (12.0 * (abs_angle - 0.30)) / 
                              (1.0 + 2.718281828 ** (12.0 * (abs_angle - 0.30))))

    # ---------- 3. lateral position penalty (encourage centering) ----------
    w_lat_pos = 0.6  # scaled up from 0.15
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
- **audit**: 所有观测均被使用，信号全面但 progress 与 centering 惩罚过弱，导致奖励几乎完全由末端接触支配，缺少快速、居中激励。
- **behavior**: agent 已学会着陆（terminated 率高，len 较长），但飞行路径可能迂回或偏离中心，环境得分提升缓慢。
- **signal**: progress_gated 幅值太小（episode sum 仅 7.7），lateral_pos_penalty 占比过低（-18），无法引导高效轨迹。
- **level**: Level 1
- **hypothesis**: 提升 progress 和 lateral 惩罚权重能使 agent 在保持成功着陆的前提下优化飞行效率与中心对准，从而显著提高环境得分。
- **risk**: 若 progress 激励过强，agent 可能因 angle gate 的抑制而用力过猛导致不稳定；lateral 惩罚增大可能引起过度修正，但当前系数仍远小于主信号，风险可控。