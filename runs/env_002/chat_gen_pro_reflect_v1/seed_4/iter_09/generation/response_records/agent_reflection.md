# Response Record

# 设计理由

当前得分为 253.03，距目标 300 分差 46.97 分。累积迭代记录显示，在将门控因子限制为纯衰减（≤1.0）后，score 和 len 同步下降，表明 upright 姿态下失去额外的速度激励是主因。本轮进行一次结构变换：**把 `gate_factor` 从 bounded [0,1] 的纯衰减改为放大‑衰减组合**，使得在身体接近直立时正向放大前进奖励（最大 1.6×），而在倾斜超过 0.25 rad 后恢复正常衰减，超过 0.5 rad 时归零。该改动恢复了之前（iter4, best=297.68）有效的放大特性，同时保留了防止摔倒的门控功能。

- **数学形式**：  
  - 当 |angle| ≤ 0.25：`gate = 1.0 + 0.6 × (1 − |angle| / 0.25)`，从 1.6 线性衰减到 1.0  
  - 当 |angle| > 0.25：`gate = max(0, 1 − (|angle| − 0.25) / 0.25)`，从 1.0 线性衰减到 0（在 0.5 rad 处）  
- **系数校准**：放大倍数 1.6 保持单组件 ≤ 2× 主信号，与历史可追索的 gate_factor 均值 ~1.6 一致；总惩罚负担不变，仍只有 0.02 的动作能量惩罚。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # Signal extraction
    # ============================================================
    horizontal_vel = obs[2]
    hull_angle = obs[0]

    # ============================================================
    # Component 1: forward_progress (linear main signal)
    # ============================================================
    forward_reward = 1.0 * max(0.0, horizontal_vel)

    # ============================================================
    # Component 2: soft_health_gate (amplify when upright, attenuate when tilted)
    #   gate ∈ [0.0, 1.6]:
    #     - |angle| ≤ 0.25: 1.6 → 1.0 (amplification)
    #     - |angle| > 0.25: 1.0 → 0.0 at 0.5 rad (attenuation)
    # ============================================================
    abs_angle = abs(hull_angle)
    if abs_angle <= 0.25:
        gate_factor = 1.0 + 0.6 * (1.0 - abs_angle / 0.25)
    else:
        gate_factor = max(0.0, 1.0 - (abs_angle - 0.25) / 0.25)

    gated_forward = forward_reward * gate_factor

    # ============================================================
    # Component 3: energy_penalty (unchanged)
    # ============================================================
    action_power = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = -0.02 * action_power

    # ============================================================
    # Total reward
    # ============================================================
    total_reward = gated_forward + energy_penalty

    components = {
        'forward_reward': forward_reward,
        'gate_factor': gate_factor,
        'gated_forward': gated_forward,
        'energy_penalty': energy_penalty,
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 当前奖励无有效 upright 激励，门控纯衰减导致直立姿态下的速度奖励不足。
- **behavior**: agent 在门控裁剪后移动速度变慢或姿态不够稳定，len 和 score 双双下降。
- **signal**: gate_factor 缺少放大，直立时无法放大速度奖励。
- **level**: Level 2（结构变换，将衰减门控改为放大‑衰减组合）
- **hypothesis**: 恢复直立时的正向放大，将在保持摔倒防护的前提下激励更快、更稳定的步态，从而恢复并超越此前 297.68 的最好成绩。
- **risk**: 放大可能略微增加能耗或倾斜风险，但门控衰减 (angle>0.25) 和能量惩罚会抑制极端行为；若 len 下降，可后续微调放大系数，但概率较低。
