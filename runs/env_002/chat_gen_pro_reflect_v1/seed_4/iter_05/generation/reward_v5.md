# 设计理由
本轮通过信号覆盖审计发现：当前 reward 函数缺少对"接近摔倒"状态的直接前兆信号——gate_factor 仅在角度大时衰减前进奖励，未在危险角度区间直接施加姿态惩罚。这导致在 hull_angle > 0.4 rad 区间缺乏垂直姿态修正压力，使 agent 在高速前进时容易失衡。由于得分已逼近 300 分但能耗尚有余地优化，本轮引入小型 hinge 角度惩罚，在角度接近摔倒边界时给予微弱但持续的修正信号，驱使关节策略更稳定、能耗更低。

**修改的组件**：新增 `angle_hinge_penalty`，当 |hull_angle| > 0.4 时启用平方惩罚，系数 0.5；角度 ≤ 0.4 时惩罚为 0 以保护安全区自由探索。此惩罚属于结构变换（缺职责 → 增组件），与已有 `gate_factor` 互补：`gate_factor` 负责在危险区衰减前进激励，`angle_hinge_penalty` 负责在危险区施加姿态修正。

**系数校准**：
- 主信号 per-step ≈ 495.4 / 1032 ≈ 0.48 (`forward_reward`)。
- 新惩罚在 |hull_angle|=0.42 时为 `0.5 * (0.02)^2 = 0.0002`，远低于 0.3×0.48 ≈ 0.14 的上限。
- 即使角度达到 0.6，惩罚也仅 0.02，与 energy_penalty（约 0.025/step）合计仍低于主信号的 0.5 倍。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # Signal extraction
    # ============================================================
    horizontal_vel = obs[2]
    hull_angle = obs[0]

    # ============================================================
    # Component 1: forward_progress (main learning signal)
    # ============================================================
    forward_reward = 1.0 * horizontal_vel

    # ============================================================
    # Component 2: soft_health_gate (safety: attenuates forward reward
    #   when balance degrades; gate = 1.0 for |angle| ≤ 0.25 rad)
    # ============================================================
    gate_lower = 0.25
    gate_upper = 0.5
    gate_raw = max(0.0, 1.0 - (abs(hull_angle) - gate_lower) / (gate_upper - gate_lower))
    gate_factor = gate_raw
    gated_forward = forward_reward * gate_factor

    # ============================================================
    # Component 3: energy_penalty (scaled to ~3% of main signal)
    # ============================================================
    action_power = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = -0.02 * action_power

    # ============================================================
    # Component 4 (NEW): angle_hinge_penalty
    #   Adds posture correction when body lean approaches fall
    #   threshold (0.5 rad). Penalty = 0.5 * (|angle| - 0.4)^2
    #   for |angle| > 0.4, else 0. This protects safe erect postures
    #   while nudging risky leans back toward stability.
    # ============================================================
    angle_threshold = 0.4
    angle_excess = max(0.0, abs(hull_angle) - angle_threshold)
    angle_hinge_penalty = -0.5 * angle_excess ** 2

    # ============================================================
    # Total reward
    # ============================================================
    total_reward = gated_forward + energy_penalty + angle_hinge_penalty

    # ============================================================
    # Components dict
    # ============================================================
    components = {
        'forward_reward': forward_reward,
        'gate_factor': gate_factor,
        'gated_forward': gated_forward,
        'energy_penalty': energy_penalty,
        'angle_hinge_penalty': angle_hinge_penalty,
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺少针对接近摔倒状态的 hinge 型角度惩罚；gate_factor 只衰减奖励不直接修复姿态。
- **behavior**: agent 高速前进但平衡余量不足，导致能耗优化尚有提升空间。
- **signal**: 缺垂直姿态修正信号（摔倒前兆的中期警告）。
- **level**: Level 2
- **hypothesis**: 在危险角度区间加入微弱但持续的平方惩罚，能提供姿态修正梯度，使步态更稳定、能耗更低，从而补齐 300 分的最后 2.3 分。
- **risk**: 惩罚系数 0.5 极小，最可能副作用是略微降低探索性跳跃步态的尝试频率，但不会抑制有效前进。