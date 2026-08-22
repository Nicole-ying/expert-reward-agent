# Response Record

# 设计理由
- **审计结论**：所有观测维度已被使用，信号覆盖齐全；终止模式显示 19/20 回合因超时（max steps）截断，agent 在垫子上方徘徊利用 C_landing_steady 的持续奖励，未能触发成功终止——这是典型的稳态奖励 exploit，需要收紧着陆条件的阈值来迫使 agent 更精确着陆并尽早稳定。
- **修改内容**：回退 iter8 对 C_landing_steady 的阈值放宽。将 `speed_factor` 的“满分”阈值从 0.3 下降到 0.15、线性衰减区间改为 [0.15, 0.35]；`angle_factor` 的阈值从 0.25 降到 0.15、衰减区间 [0.15, 0.35]；`angular_factor` 的阈值从 0.8 降到 0.5、衰减区间 [0.5, 1.0]；`dist_factor` 的有效范围从 0.25 收紧到 0.2。这些调整使 C_landing_steady 只有在更逼近理想着陆条件时才产生奖励，从根本上削弱 agent“占据次优稳态持续获奖”的 exploit，同时保留清晰的梯度引导其向精确软着陆收敛。
- **系数校准**：C_landing 的基础系数保持 0.15。收紧阈值后，C_landing 的 active_rate 和 per-step reward 都会轻微下降，但与 A_progress_gated 的比例仍合理，且不会引入额外惩罚负担。

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

    scaled_progress = potential_delta * 10.0
    gated_progress = scaled_progress * gate

    # --- Component C: landing steady-state reward (TIGHT thresholds, revert iter8 relaxations) ---
    # Distance factor: linear decay, zero beyond 0.2 (tightened from 0.25)
    dist_factor = max(0.0, 1.0 - distance_to_target / 0.2)

    # Contact: both legs
    contact_factor = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0

    # Speed factor: full score below 0.15, linear decay to 0 at 0.35 (tightened from 0.3/0.5)
    if speed < 0.15:
        speed_factor = 1.0
    else:
        speed_factor = max(0.0, 1.0 - (speed - 0.15) / 0.2)

    # Angle factor: full score below 0.15, decay to 0 at 0.35 (tightened from 0.25/0.45)
    if abs(angle) < 0.15:
        angle_factor = 1.0
    else:
        angle_factor = max(0.0, 1.0 - (abs(angle) - 0.15) / 0.2)

    # Angular velocity factor: full score below 0.5, decay to 0 at 1.0 (tightened from 0.8/1.2)
    if abs(angular_vel) < 0.5:
        angular_factor = 1.0
    else:
        angular_factor = max(0.0, 1.0 - (abs(angular_vel) - 0.5) / 0.5)

    landing_factor = dist_factor * contact_factor * speed_factor * angle_factor * angular_factor
    C_landing = 0.15 * landing_factor   # coefficient unchanged

    total_reward = gated_progress + C_landing

    components = {
        'A_progress_gated': gated_progress,
        'C_landing_steady': C_landing
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 所有观测已被使用，无信号缺口；19/20 truncated 表明 agent 在垫子上方维持次优稳态而不终止，属于典型稳态奖励 exploit。
- **behavior**: agent 在着陆垫附近徘徊利用 C_landing_steady 的持续奖励（active_rate≈79%），未能尽早收敛到成功终止条件。
- **signal**: C_landing_steady 的宽松阈值使得“占据好状态即持续获奖”，削弱了向精确软着陆收敛的动力。
- **level**: Level 1
- **hypothesis**: 收紧着陆条件阈值（速度/角度/角速度/距离门槛）将迫使 agent 追求更精确的着陆姿态，从而降低 exploit 收益、加速稳定并触发终止，有望恢复分数上升趋势。
- **risk**: 收紧过于激进可能导致 landing 探索梯度消失，active_rate 骤降。如果下一轮连续 ❌，需转为结构变换（如改善量奖励）。
