# Response Record

# 设计理由

### 第 0 步：信号覆盖审计
- **终止 → 前兆**：#5.1 终止条件：高度越界 (z<0.2 or z>1.0)、NaN/inf。代码有 height_survival_penalty，阈值 low_thresh=0.35 太保守（留了 43% 的死亡地带无梯度）。**没有侧向漂移或角速度前兆信号**——很多跌倒始于侧倾/旋转而非直接触底。
- **目标 → 进度**：#5.1 核心任务是保持身体竖直并稳定向前行走。代码 forward_upright_product 直接处理这个目标——✅ 已覆盖。
- **效率信号**：动作维度 8 ≥ 6，有 torque_penalty——✅ 已覆盖。
- **僵尸组件**：height_survival_penalty active_rate=16.9%。不是僵尸（边界足够），但处罚方式可优化——它只是在事情已经变糟后处罚，而无法在接近边界时渐进地抑制不良行为。
- **一句话结论**：当前 reward 漏了侧向稳定性（body_y_velocity, angular velocity）信号，高度约束离死亡边界太远太弱，导致机器人快速侧倾/旋转死亡（terminated=19/20）。

### 第 1 步：行为诊断
1. **agent 在做什么？** 快速失败。terminated=19/20，len=349 远早于 1000。score 从 -544 到 -49 改善，但生存时间从 648 降到 349——这是 torch_penalty 抑制了维持平衡所需的核心动作吗？可能。但更根本的是：主信号 forward_upright_product 只驱使前向，对侧倾、旋转完全无视，机器人一边试图加速前进一边侧向失衡死亡。

2. **干预哪个目标？** 侧向稳定性。解决快速失败必须先阻止致命跌倒。

3. **方向还值得继续吗？** iter 2 添加动作惩罚 — 预期改善终止率，实际终止率恶化（len 从 648 降到 349）。这个方向（加独立惩罚）可能在治标。但只有两轮，且 score 正在快速上升——方向值得继续，但需要**改进信号质量而非再加独立惩罚**。

### 第 2 步：选择干预层级 — Level 2（结构变换）

**证据与变换对应关系**：
- **height_survival_penalty 独立惩罚 → soft_health_gate**：terminated 主因是越界，惩罚已加但 inactive_rate 太高（仅 16.9% 触发），证明阈值离死亡边界太远。将该状态做成 gate 乘到主奖励上，让恶化渐进地减小奖励而非只在已越界时处罚。
- **add 新组件 — 侧向稳定性**：信号缺口明显（y_velocity, angular velocity 全无使用）。添加有界软约束。

### 设计校准
1. 主信号 per-step ≈ forward_upright_product / len = 366.6 / 349.4 ≈ 1.05
2. **height_gate**：hinge 设在 0.3（死亡边界 0.2 的 150%，因 len 尚可接受，不应堆砌过多门槛）。中段（0.4-0.8）gate ≈ 1.0，确保“正常生存”不被惩罚。
3. **lateral_penalty**：|body_y_velocity| 惩罚，per-step 目标 ≤ 0.15 ≤ 0.15x 主信号。系数设为 0.5，hinge 在 0.15。正常行走可能产生 ~0.2-0.3 的侧滑，将产生约 0.025-0.075 的惩罚——可接受。
4. **rotation_penalty**：角速度惩罚，per-step ≤ 0.1。系数 0.2，hinge 在 0.5。
5. 总惩罚负担：0.5 + 0.1 ≈ 0.6 → 略高但尚在 0.5x 边界处。鉴于 terminated 率高，略强的稳定性约束是必要的。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract key signals from observation (current obs, not next_obs)
    body_z = obs[0]
    qx = obs[2]
    qy = obs[3]
    body_x_vel = obs[13]
    body_y_vel = obs[14]
    body_roll_vel = obs[16]
    body_pitch_vel = obs[17]
    body_yaw_vel = obs[18]

    # ----- uprightness (projection of body z-axis onto world z-axis) -----
    body_up_z = 1.0 - 2.0 * (qx**2 + qy**2)
    body_up_z = max(0.0, min(1.0, body_up_z))  # numerical safety clamp

    # ----- core learning signal: forward velocity modulated by uprightness -----
    forward_reward = body_x_vel * body_up_z

    # ----- height survival gate (continuous, bounded [0, 1]) -----
    # Transform from independent penalty to multiplicative gate.
    # Safe zone: [0.4, 0.8] → gate ≈ 1.0
    # Warning zones: [0.3, 0.4] and [0.8, 0.9] → gate ramps linearly to 0
    # Death zones: <0.3 or >0.9 → gate = 0
    z_low_safe = 0.4
    z_low_danger = 0.3
    z_high_safe = 0.8
    z_high_danger = 0.9

    if body_z >= z_low_danger and body_z <= z_low_safe:
        height_gate = (body_z - z_low_danger) / (z_low_safe - z_low_danger)  # 0.0 → 1.0
    elif body_z >= z_low_safe and body_z <= z_high_safe:
        height_gate = 1.0
    elif body_z >= z_high_safe and body_z <= z_high_danger:
        height_gate = (z_high_danger - body_z) / (z_high_danger - z_high_safe)  # 1.0 → 0.0
    else:
        height_gate = 0.0

    # Apply gate to forward reward — bad height suppresses all forward credit
    gated_forward = forward_reward * height_gate

    # ----- lateral stability penalty (hinge on |body_y_velocity|) -----
    lateral_threshold = 0.15  # minor lateral drift is acceptable
    lateral_error = max(0.0, abs(body_y_vel) - lateral_threshold)
    w_lateral = 0.5
    lateral_penalty = -w_lateral * lateral_error

    # ----- rotation stability penalty (hinge on angular velocities) -----
    # Penalize excessive rotation about any axis
    rotation_threshold = 0.5
    roll_error = max(0.0, abs(body_roll_vel) - rotation_threshold)
    pitch_error = max(0.0, abs(body_pitch_vel) - rotation_threshold)
    yaw_error = max(0.0, abs(body_yaw_vel) - rotation_threshold)
    w_rotation = 0.2
    rotation_penalty = -w_rotation * (roll_error + pitch_error + yaw_error)

    # ----- action penalty for energy efficiency (unchanged) -----
    w_action = 0.03
    torque_penalty = -w_action * sum(action**2) / 8.0

    total_reward = gated_forward + lateral_penalty + rotation_penalty + torque_penalty

    components = {
        "forward_upright_gated": gated_forward,
        "height_gate_value": height_gate,  # diagnostic
        "lateral_penalty": lateral_penalty,
        "rotation_penalty": rotation_penalty,
        "torque_penalty": torque_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 侧向漂移和角速度完全无信号覆盖，高度惩罚离死亡边界太远（16.9% 触发率），导致机器人在有充足时间改善前就侧倾死亡。
- **behavior**: agent 快速失败（terminated 19/20, len≈349），正试图加速前进来提高 forward_upright_product，但无任何信号抑制侧向失衡。
- **signal**: 缺侧向稳定性约束和渐进式高位门槛。现有的 height_survival_penalty 作为独立项太弱，需要在恶化时直接削弱主信号的奖励梯量。
- **level**: Level 2
- **hypothesis**: 将高度约束改为 gate 将迫使 agent 在高度不良时接受近乎零的前进奖励，打破「加速→失衡→死亡」的循环。添加侧向和旋转惩罚填补信号缺口，提供早期纠正梯度。这三项合力应显著降低 terminated 率，允许 agent 存活到学会稳定步态。
- **risk**: 过度约束稳定性可能使 agent 变得过于保守，牺牲前进速度。如果 terminated 率改善但 score 增长停滞，下一轮需调降 lateral/rotation 系数或提高阈值。
