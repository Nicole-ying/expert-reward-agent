# 设计理由

**改了什么组件**：将 `forward_reward` 从单纯的门控前向速度 `v_x * height_gate` 改为同时受高度和直立姿态双重门控的 `v_x * height_gate * upright_gate`。  
同时将 `upright_penalty` 的权重 `w_up_pen` 设为 `0.0`（实际即移除该惩罚），因为其职责已被直立门控内化——直立姿态的好坏直接调制前向奖励的密度，无需额外的独立惩罚（其 `active_rate = 5.5%` 也表明它几乎没有起作用）。

**数学形式**  
- `up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)`（已有）  
- `upright_gate = max(0.0, up_z)`：当身体完全倒立时门控为 0，完全直立时为 1，连续、有界、不塌缩。  
- `forward_reward = w_fwd * v_x * height_gate * upright_gate`  
  其中 `w_fwd = 1.0`，`height_gate` 保持原设计。  
- `upright_penalty` 不再参与奖励计算。

**系数校准**  
- 保持 `w_fwd = 1.0`。当前 `forward_reward` 的每步均值约为 `1.8`（估计），加入 `upright_gate` 后，若 `up_z ≈ 0.9`，信号下降约 10%，仍在可接受范围。  
- `lateral_penalty`（每步 `-0.1`）与 `action_penalty`（`-0.003`）合计远低于主信号的 0.5 倍，符合惩罚负担要求。  
- 不做其他系数变动。

**与 iter3 成功骨架的关系**  
历史最佳 iter3 使用 `gated_forward`（极可能也是 `v_x * height_gate * upright_gate`）并配以 `upright_bonus`，得到了 1839 分。当前版本关闭了直立门控并改用独立惩罚，导致连续 4 轮大幅落后。本次修改恢复直立与前向的耦合，是回归已验证路径的最小可验证步骤。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- signal extraction ----
    body_z   = obs[0]
    quat_x   = obs[2]
    quat_y   = obs[3]
    v_x      = obs[13]
    v_y      = obs[14]

    # ---- upright measure (1.0 when perfectly upright) ----
    up_z = 1.0 - 2.0 * (quat_x ** 2 + quat_y ** 2)

    # ---- height gate: 1.0 in safe zone, decays to 0 near hard limits ----
    z_low  = 0.25   # termination boundary: z <= 0.2
    z_high = 0.95   # termination boundary: z >= 1.0
    z_safe_low  = 0.35
    z_safe_high = 0.85

    low_factor = (body_z - z_low) / (z_safe_low - z_low)
    low_factor = max(0.0, min(1.0, low_factor))

    high_factor = (z_high - body_z) / (z_high - z_safe_high)
    high_factor = max(0.0, min(1.0, high_factor))

    height_gate = low_factor * high_factor

    # ---- upright gate: continuous, 1 when fully upright, 0 when horizontal or inverted ----
    upright_gate = max(0.0, up_z)

    # ---- forward progress (main signal) ----
    #   Gating: must stay within safe height AND remain upright to earn forward reward.
    w_fwd = 1.0
    forward_reward = w_fwd * v_x * height_gate * upright_gate

    # ---- upright penalty: disabled (role taken over by upright_gate) ----
    #   Weight set to 0.0, component kept for logging compatibility.
    upright_threshold = 0.7
    upright_deficit = max(0.0, upright_threshold - up_z)
    w_up_pen = 0.0
    upright_penalty = -w_up_pen * upright_deficit

    # ---- lateral stability (quadratic penalty) ----
    w_lat = 0.3
    lateral_penalty = -w_lat * (v_y ** 2)

    # ---- action smoothness ----
    w_act = 0.005
    action_penalty = -w_act * sum(a ** 2 for a in action) / len(action)

    # ---- total reward ----
    total_reward = forward_reward + upright_penalty + lateral_penalty + action_penalty

    components = {
        "forward_reward":   forward_reward,
        "upright_penalty":  upright_penalty,
        "lateral_penalty":  lateral_penalty,
        "action_penalty":   action_penalty,
        "_height_gate":     height_gate
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺少对终止高度边界（z≤0.2 / z≥1.0）的 explicit 接近惩罚，仅靠 gate 衰减；直立组件僵尸（active 5.5%），但当前最紧迫的结构缺陷是前向信号与直立解耦。
- **behavior**: agent 在 iter8 中解除了直立约束，用弱惩罚代替，导致前向奖励虽高（~1633）但实际总分仅 464（可能与隐藏的步态低效或评估差异有关），远未恢复 iter3 水平。
- **signal**: 前向奖励缺乏直立反馈；独立直立惩罚几乎不触发，无法提供有效梯度。
- **level**: Level 2 — 结构变换（将前向奖励改为双重门控并移除僵尸惩罚）
- **hypothesis**: 将前向收益与直立姿态强耦合（`forward_reward = v_x * height_gate * upright_gate`）能迫使 agent 在追求速度时保持直立，从而复制 iter3 的高分模式。
- **risk**: 若 `upright_gate` 在初始阶段长期偏低，前向信号会减弱，可能延长探索时间；但由于并非从头训练，合理降低初始惩罚负担有助于恢复。