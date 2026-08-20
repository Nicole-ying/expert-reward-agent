# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract necessary observations
    body_z = obs[0]
    qw = obs[1]
    qx = obs[2]
    qy = obs[3]
    qz = obs[4]
    body_x_velocity = obs[13]
    body_y_velocity = obs[14]
    body_roll_velocity = obs[16]
    body_pitch_velocity = obs[17]

    # Compute body up vector z-component (uprightness)
    body_up_z = 1.0 - 2.0 * (qx**2 + qy**2)
    body_up_z = max(-1.0, min(1.0, body_up_z))  # numerical safety

    # Health gate: height
    height_low = 0.2
    height_low_margin = 0.15   # 0.2 + 0.15 = 0.35
    height_high = 0.95
    height_high_margin = 0.05  # 0.95 - 0.05 = 0.90

    # Height gate: linear decay to 0 when approaching boundaries
    if body_z < height_low:
        height_gate = 0.0
    elif body_z < height_low + height_low_margin:
        height_gate = (body_z - height_low) / height_low_margin
    elif body_z > height_high:
        height_gate = 0.0
    elif body_z > height_high - height_high_margin:
        height_gate = (height_high - body_z) / height_high_margin
    else:
        height_gate = 1.0

    # Uprightness gate: linear decay when too tilted
    upright_threshold = 0.2   # body_up_z below this -> gate 0
    upright_full = 0.5        # body_up_z above this -> gate 1
    if body_up_z < upright_threshold:
        upright_gate = 0.0
    elif body_up_z < upright_full:
        upright_gate = (body_up_z - upright_threshold) / (upright_full - upright_threshold)
    else:
        upright_gate = 1.0

    gate = height_gate * upright_gate

    # Forward progress (main signal)
    w_forward = 1.0
    forward_progress = w_forward * max(0.0, body_x_velocity) * gate  # only reward positive x velocity

    # Lateral drift penalty (hinge)
    w_lat = 0.5
    lat_drift_margin = 0.3 * 0.0  # if body_y_velocity is in [-0.3, 0.3], no penalty
    lateral_drift_penalty = -w_lat * max(0.0, abs(body_y_velocity) - 0.3)

    # Roll/pitch stability hinge
    w_rp = 0.2
    ang_vel_margin = 1.0  # rad/s
    roll_penalty = max(0.0, abs(body_roll_velocity) - ang_vel_margin)
    pitch_penalty = max(0.0, abs(body_pitch_velocity) - ang_vel_margin)
    roll_pitch_penalty = -w_rp * (roll_penalty + pitch_penalty)

    total_reward = forward_progress + lateral_drift_penalty + roll_pitch_penalty

    components = {
        'forward_progress': forward_progress,
        'lateral_drift_hinge': lateral_drift_penalty,
        'roll_pitch_hinge': roll_pitch_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务画像
- **task_family**: locomotion_continuous_control
- **dynamics_subtype**: multi_legged_body_locomotion (四足机器人)
- **核心目标**: 沿世界 x 轴持续稳定前进，不能以牺牲稳定性为代价。
- **失败风险**: 身体过低（摔倒）、过高（弹跳）、严重倾斜或仿真溢出。

## 2. 选用的奖励职责（roles）
依据 `environment_card.md` 中的 `reward_role_decomposition`：

- **forward_velocity**（主学习信号）  
  - 作用：驱动机器人产生正向 x 轴速度。  
  - 信号：`body_x_velocity (obs[13])`。  
  - 公式算子：`dense_state_signal` 正线性奖励 `w * max(0, signal)`，只奖励正向速度，不奖励后退。

- **health_gate**（健康门控）  
  - 作用：当身体高度或直立程度恶化时，按比例削弱前进奖励，防止“先冲后摔”。  
  - 信号：`body_z (obs[0])` 和由四元数导出的 `body_up_z`。  
  - 公式算子：`soft_health_gate`，使用两个线性衰减因子相乘。  
    - 高度门：在危险区域 `[0.2, 0.35]` 和 `[0.90, 0.95]` 线性衰减。  
    - 直立门：`body_up_z` 在 `[0.2, 0.5]` 线性衰减（低于 0.2 时门关闭）。  
  - 注意：门不是独立加法项，它通过乘法进入 `forward_progress`，不单独记录为加法 component（但 `forward_progress` 已经包含了门控效果）。

- **lateral_drift_penalty**（条件约束）  
  - 作用：抑制侧向漂移，使前进更高效。  
  - 信号：`body_y_velocity (obs[14])`。  
  - 公式算子：`dense_state_signal` 的 hinge 形式 `-w * max(0, |signal| - margin)`，margin = 0.3 m/s。在安全范围内不罚，避免干扰正常步态中的微小侧向摆动。

- **roll_pitch_stability_penalty**（条件约束）  
  - 作用：限制过大角速度，降低倒地风险。  
  - 信号：`body_roll_velocity (obs[16])`, `body_pitch_velocity (obs[17])`。  
  - 公式算子：`dense_state_signal` hinge 形式，角速度 margin = 1.0 rad/s，超出部分累加后乘以小权重。

**v1 共使用 3 个加法组件**（forward_progress + 两个 hinge 惩罚），符合角色预算（2~4），且优先覆盖主要学习信号和安全约束。

## 3. 排除的职责及原因
- **action_cost**（能量效率）：任务未明确要求节能，且四足行走需要较大扭矩，v1 不引入动作代价，避免压制步态探索。
- **terminal_success_bonus**：环境无到达终点概念，success flag 不存在，无法使用。
- **terminal_failure_penalty**：无显式 failure flag；在终止步观测可能已损坏（NaN/Inf），不宜使用。通过 health gate 的前兆衰减已提供软惩罚。
- **uprightness_penalty、vertical_stability_penalty**：服从 gate 的软约束已覆盖这些健康考量，不再重复。
- **action_smoothness**：需要历史动作信息（当前接口不提供），且 v1 暂不关注关节平滑，留待后续。
- **curriculum_weighting**：training_progress 可用但 v1 无需动态权重，保持简单即可。

## 4. 为什么没有使用 terminal_success_reward / terminal_failure_penalty
- 环境卡片明确设定 `explicit_success_flag_available=false`，且不存在任何 success termination。
- `explicit_failure_flag_available=false`，无法可靠区分失败类型；即使推测失败也不应该从可能损坏的 `next_obs` 中发放大额惩罚。
- 替代方案：health gate 在健康恶化时连续衰减主信号，实现“前兆软约束”，比一次性的 terminal 惩罚更稠密、更安全。

## 5. 留待后续迭代的职责
- 动作平滑性 / 扭矩效率（当出现震动或能耗问题时加入）。
- 接触相关信号（若未来提供足端接触信息，可加入 foot clearance 或 contact pattern 奖励）。
- 动态课程或速度目标（当前无目标速度参考，若任务演化出速度区间需求可加入）。
- 更强的摔倒预测与恢复引导（可加入更精细的姿态能量项）。

## 6. 训练后应观察的 failure modes
1. **agent 静止不动**：gate = 1 但前进速度为 0，说明探索不足或主奖励线性梯度太弱。后续可考虑将 `body_x_velocity` 改为平方或指数形式，提供更大梯度。
2. **agent 以极慢速度稳定行走**：需要检查是否因为 hinge 惩罚轻微侧向或角速度抑制了正常步态振幅，可适当放宽 margin 或降低权重。
3. **agent 在 gate 衰减区反复徘徊**：gate 在 0.6 附近来回波动，导致主奖励不稳定。若观察到，可扩大 safe zone 或采用更平滑的衰减函数（如 Sigmoid 型）。
4. **agent 学会弹跳前进**：health gate 的高度上界会部分抑制，但当弹跳很短暂时仍可能漏过；后续可加入 `body_z_velocity` 的震荡惩罚。
5. **未形成有效步态，仅短促爆发然后倒下**：gate 已限制主奖励，但 hinge 惩罚可能在摔倒前几帧才增大；可考虑加入更早期的姿态预警。