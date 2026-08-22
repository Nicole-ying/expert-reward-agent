# 设计理由
本轮修改针对组件 **A_progress_gated** 的缩放因子：从 `potential_delta * 10.0` 提升到 `potential_delta * 25.0`。

**诊断逻辑**：
- 信号覆盖审计：所有8个观测均已被使用，无信号缺失；终止模式为20/20截断，说明 agent 已能长期存活并着陆于垫上，但整体得分卡在约145分。
- 行为诊断：agent 能着陆，但 `A_progress_gated` 的 per‑step 贡献仅约 0.0056，远小于 `C_landing_steady` 的约 0.099。弱 progress 信号导致下降过程缺乏足够的梯度压力，agent 可能在前期徘徊较久，压缩了后期累积 landing 奖励的步数（episode 固定截断 1000 步）。
- 累积记录显示同一骨架（`A_progress_gated + C_landing_steady`）已迭代4轮且最近一轮收紧阈值有效，说明骨架方向正确，但需要强化下降引导以缩短抵达时间，从而在有限步数内收获更多 landing 奖励。
- 选择 **Level 2 结构变换**（信号尺度调整）：提升 progress 缩放因子，但不改变数学形式或其与其他组件的耦合。

**数学形式**：`scaled_progress = potential_delta * 25.0`，其余 gate 逻辑保持不变。由于 gate 在低速/着陆后自动释放为 1.0（当前 active_rate 100% 表示无负面压榨），放大后的 progress 不会破坏安全约束。计算结果：per‑step progress 将升至约 0.014（0.0056×2.5），仍远低于 landing 的 0.1，符合总惩罚/奖励负担约束（主信号 per‑step 0.099，progress 占比 < 0.3x）。

**系数校准**：保持 `gate` 的 `safe_speed = 0.3 + 1.5 * distance_to_target` 不变，以免引入新的约束冲突。唯一的变量是 `potential_delta` 的放大系数，从 10 增到 25。

**预期**：更强的下降梯度会促使 agent 更快接近着陆垫，留下更多步数在垫上获得 `C_landing_steady` 奖励，从而提升 episode 总分（目标从 ~146 推进至 160+）。不预期 episode 长度变化（仍为 1000）。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- Extract next_obs signals ---
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- Helper: distance from pad center ---
    horizontal_dist = abs(x)
    distance_to_target = (horizontal_dist**2 + y**2) ** 0.5

    # --- Component A: main progress signal via potential-based shaping ---
    norm_distance = distance_to_target / 2.5
    angle_penalty = abs(angle) / 1.57
    potential = -(norm_distance + 0.3 * angle_penalty)

    prev_x = obs[0]
    prev_y = obs[1]
    prev_angle = obs[4]
    prev_horizontal_dist = abs(prev_x)
    prev_distance = (prev_horizontal_dist**2 + prev_y**2) ** 0.5
    prev_norm_distance = prev_distance / 2.5
    prev_angle_penalty = abs(prev_angle) / 1.57
    prev_potential = -(prev_norm_distance + 0.3 * prev_angle_penalty)

    potential_delta = potential - prev_potential

    # --- Component B: soft velocity health gate (unchanged) ---
    speed = (vx**2 + vy**2) ** 0.5
    safe_speed = 0.3 + 1.5 * distance_to_target
    overspeed_ratio = speed / (safe_speed + 1e-6)
    gate = 0.3 + 0.7 * (2.718281828 ** (-max(0.0, overspeed_ratio - 1.0)))

    # *** Modified scaling factor: 10.0 → 25.0 ***
    scaled_progress = potential_delta * 25.0
    gated_progress = scaled_progress * gate

    # --- Component C: landing steady-state reward (unchanged) ---
    dist_factor = max(0.0, 1.0 - distance_to_target / 0.2)
    contact_factor = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0

    if speed < 0.15:
        speed_factor = 1.0
    else:
        speed_factor = max(0.0, 1.0 - (speed - 0.15) / 0.2)

    if abs(angle) < 0.15:
        angle_factor = 1.0
    else:
        angle_factor = max(0.0, 1.0 - (abs(angle) - 0.15) / 0.2)

    if abs(angular_vel) < 0.5:
        angular_factor = 1.0
    else:
        angular_factor = max(0.0, 1.0 - (abs(angular_vel) - 0.5) / 0.5)

    landing_factor = dist_factor * contact_factor * speed_factor * angle_factor * angular_factor
    C_landing = 0.15 * landing_factor

    total_reward = gated_progress + C_landing

    components = {
        'A_progress_gated': gated_progress,
        'C_landing_steady': C_landing
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 所有观测已覆盖，无信号缺失；agent 截断为主，已会着陆，但 progress 信号过弱导致前期徘徊，浪费可积累 landing 奖励的步数。
- **behavior**: agent 存活满 1000 步并在着陆垫上获取持续的 landing 奖励，但下降阶段太慢，错过更早着陆的机会。
- **signal**: progress shaping 的 per‑step 贡献极弱（~0.0056），只提供微弱引导，与 landing 主信号（~0.099）严重不对等。
- **level**: Level 2
- **hypothesis**: 增强 progress 缩放会提供更强的梯度鼓励 agent 快速接近目标，从而提前着陆，增加截断前积累的高额 landing 奖励，总分将提高。
- **risk**: 若下降过快，gate 可能会持续压低奖励，但 gate 当前 active_rate 100% 表明其只在必要时段生效且不阻挡正常低速行为，风险可控。