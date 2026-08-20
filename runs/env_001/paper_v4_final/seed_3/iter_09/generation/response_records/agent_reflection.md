# Response Record

# 设计理由
本轮上一版（iter 8）把 `contact_landing_reward` 的指数从 0.25 提升到 0.5，预期强化“完美着陆”信号。实际却导致 episode 长度从 1000 断崖暴跌至 69.8，所有 episode 均为 terminated（非超时），说明 agent 在早期就触发了环境终止条件（大概率是姿态角越界），根本等不到接触着陆垫。**信号校准缺口在于：当前奖励对角度越界没有任何直接的负反馈——`angle_gate` 只是削弱 `progress`，不构成惩罚梯度；`angvel_penalty` 只能抑制角速度突变，不能阻止 agent 长时间保持一个大倾角下滑。**  
因此本轮**添加一个 hinge 形式的终端临界角度惩罚**：当 `|angle|` 进入边界附近（阈值 0.7 rad，约为终止边界的 60‑80%）时施加二次惩罚，给予 agent 明确的“远离危险区”信号。该组件只在角度激进时触发，正常飞行时 active rate 低，不会形成难以承受的惩罚负担，且不会像全局 penalty 那样在早期压制必要探索。  
同时保留现有 `contact_landing_reward`（指数 0.5）、`angle_gate` 乘法机制，完全不同于 iter 7 的 `**0.25` 版本，构成一次切实的结构补充。

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
    # sigmoid with steepness 12, center 0.30
    angle_gate = 1.0 - 0.8 * (2.718281828 ** (12.0 * (abs_angle - 0.30)) / 
                              (1.0 + 2.718281828 ** (12.0 * (abs_angle - 0.30))))

    # ---------- 3. lateral position penalty (encourage centering) ----------
    w_lat_pos = 0.08
    lateral_pos_penalty = -w_lat_pos * (nx ** 2)

    # ---------- 4. angular velocity penalty (smooth rotation) ----------
    w_angvel = 0.05
    angvel_penalty = -w_angvel * (n_ang_vel ** 2)

    # ---------- 5. contact-based landing proxy (exponent 0.5) ----------
    mean_contact = (n_lc + n_rc) / 2.0

    k_y = 10.0
    k_vy = 8.0
    k_ang = 15.0
    f_y   = 1.0 / (1.0 + k_y   * abs(ny))
    f_vy  = 1.0 / (1.0 + k_vy  * abs(nvy))
    f_ang = 1.0 / (1.0 + k_ang * abs(n_angle))

    contact_landing_factor = (mean_contact * f_y * f_vy * f_ang) ** 0.5
    w_contact_land = 5.0
    contact_landing_reward = w_contact_land * contact_landing_factor

    # ---------- 6. NEW: angle-terminal-proximity penalty ----------
    # Hinge penalty activates when angle approaches a dangerous region
    # (estimated termination threshold around 1.0 rad; start warning at 0.7)
    angle_term_threshold = 0.7
    w_angle_term = 1.5
    angle_term_penalty = -w_angle_term * (max(0.0, abs_angle - angle_term_threshold) ** 2)

    # ---------- combine ----------
    total_reward = (w_progress * progress * angle_gate
                    + lateral_pos_penalty
                    + angvel_penalty
                    + contact_landing_reward
                    + angle_term_penalty)

    components = {
        "progress_gated": w_progress * progress * angle_gate,
        "lateral_pos_penalty": lateral_pos_penalty,
        "angvel_penalty": angvel_penalty,
        "contact_landing_reward": contact_landing_reward,
        "angle_term_penalty": angle_term_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 所有观测维度已被使用，信号缺失不成立；核心问题是 **终止临界区缺乏负反馈**，agent 在无预警的情况下冲入角度危险区直接触发生效终止。
- **behavior**: agent 在 70 步内因姿态角越界触发生效终止，等不到接触着陆垫。
- **signal**: 缺少针对角度接近终端阈值的先行惩罚——`angle_gate` 只削弱进步，不形成逃避危险区的梯度。
- **level**: Level 2
- **hypothesis**: 新增 hinge 惩罚会让 agent 在角度接近终止边界时收到直接的负梯度，主动压低角度，从而延长 episode 并有机会完成着陆。
- **risk**: 若阈值设置过于接近实际终止线（实际终止可能比 1.0 rad 更早），agent 可能仍会在得到惩罚前就终止；可通过观察 `angle_term_penalty` 的 active_rate 判断，若仍 0% 可下调 `angle_term_threshold`。
