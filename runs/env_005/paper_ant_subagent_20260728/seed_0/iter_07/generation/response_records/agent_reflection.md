# Response Record

# 设计理由
本轮诊断基于审计发现：高度惩罚 active_rate 极低，但终止还在发生（25% terminated），说明高度安全约束并未有效转化为梯度。`lateral_penalty` 目前每步仅约 -0.015，远弱于历史最佳 iter3 的 -0.103。这种极弱的侧向约束放任 agent 产生大量侧向速度，消耗前向动能，是导致总 reward 虽为正但外部分数大幅恶化、len 维持但前进效率低下的关键。

干预层级定为 **Level 1 — 尺度修复**：保持现有二次惩罚形式不变，仅将横向惩罚系数 `w_lat` 从 0.05 提高到 0.30，使每步期望惩罚回升到约 -0.10，接近 iter3 水平。这不会改变职责结构，也不会引入新阈值 discontinuity，是对历史成功骨架的温和校准。

# 修改后的奖励函数
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- signal extraction ----
    body_z   = obs[0]
    quat_x   = obs[2]
    quat_y   = obs[3]
    v_x      = obs[13]
    v_y      = obs[14]

    # ---- upright measure (continuous, 1.0 when perfectly upright) ----
    up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---- upright gate (soft gating of forward progress) ----
    gate_lower = 0.3
    gate = (up_z - gate_lower) / (1.0 - gate_lower)
    gate = max(0.0, min(1.0, gate))

    # ---- gated forward progress (main progress signal) ----
    w_fwd = 1.0
    gated_forward = w_fwd * v_x * gate

    # ---- upright bonus (direct posture reward, increased weight) ----
    w_up = 0.5
    upright_bonus = w_up * up_z

    # ---- body height safety (soft hinge quadratic) ----
    z_low_safe  = 0.3
    z_high_safe = 0.8
    w_h = 0.5
    low_hinge  = max(0.0, z_low_safe - body_z)
    high_hinge = max(0.0, body_z - z_high_safe)
    height_penalty = -w_h * (low_hinge**2 + high_hinge**2)

    # ---- lateral stability (mild quadratic, calibrated to ~0.1 per-step) ----
    w_lat = 0.3
    lateral_penalty = -w_lat * (v_y ** 2)

    # ---- action smoothing (light energy regularizer) ----
    w_act = 0.005
    action_penalty = -w_act * sum(a**2 for a in action) / len(action)

    # ---- total reward ----
    total_reward = (gated_forward + upright_bonus +
                    height_penalty + lateral_penalty +
                    action_penalty)

    components = {
        "gated_forward":      gated_forward,
        "upright_bonus":      upright_bonus,
        "height_penalty":     height_penalty,
        "lateral_penalty":    lateral_penalty,
        "action_penalty":     action_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 高度终止前兆通过高度惩罚和直立门控覆盖，但高度惩罚几乎无梯度；当前最大信号缺口是过弱的侧向约束。
- **behavior**: agent 能存活较长（len 837）但大部分前进速度被浪费在侧向漂移上，导致外部分数极低。
- **signal**: `lateral_penalty` 幅度仅为历史最佳水平的 1/7，无法有效抑制侧向速度。
- **level**: Level 1 — 尺度修复
- **hypothesis**: 通过将 `w_lat` 从 0.05 提升到 0.30，使每步横向惩罚恢复至约 -0.10，agent 会被迫保持更纯的前向移动，从而在相同步数内积累更多有效前进距离，分数应趋近 iter3 最佳值。
- **risk**: 过强的侧向约束可能略微降低步态的探索灵活性，但由于惩罚量级仍只占主信号的 ~11%，不会主导优化方向。
