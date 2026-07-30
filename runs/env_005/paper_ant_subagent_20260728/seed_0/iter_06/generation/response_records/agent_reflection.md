# Response Record

# 设计理由
历史记忆显示，iter3 的 `gated_forward + upright_bonus` 骨架达到了接近目标的 1839 分，但后续 iter4 / iter5 试图通过 hinge 阈值修正侧向和角速度惩罚反而导致分数暴跌并始终无法恢复。根本原因不是惩罚的“量”不够，而是 hinge 形式在姿态边界处引入了不连续的梯度结构，干扰了 agent 同时学习直立门控和前向速度的策略——当 agent 尚未稳定直立时，hinge 在阈值附近的阶跃变化使得 gate 信号和惩罚信号相互矛盾，导致探索陷入死区。

因此，本轮重建以 iter3 的数学骨架为基础（保留 gate + upright_bonus + height_penalty），但采用更温和的惩罚形式：将横向和角速度惩罚恢复为**无界二次惩罚**（系数大幅降低），完全移除 hinge 阈值。这样避免了阈值引入的梯度断崖，让 agent 在任何姿态下都能获得平滑的“越小越好”的引导，不会因为短暂超限而丧失整个正向信号。同时，将 upright_bonus 权重从 0.2 提升到 0.5，更直接地强化直立姿态，使 gate 更早稳定，从而带动主前向奖励的恢复。整体框架与 iter4/iter5 的 hinge 迭代有本质不同：不是通过“超过某一界限才惩罚”来约束，而是通过“始终温和抑制＋强直立正向引导”来维持平衡。

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

    # ---- lateral stability (mild quadratic, avoids threshold discontinuities) ----
    w_lat = 0.05
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
- **audit**: 终止条件高度越界已有前兆 soft hinge，主目标前进速度有门控奖励且配直立奖励，缺角速度抑制可能导致翻滚但暂时移除以免干扰；无僵尸组件。
- **behavior**: agent 在 iter5 中维持了 700 步但仍频繁终止，主信号 gated_forward 已形成主导但总分仍为负，说明 gate 在部分回合完全关闭或横向/角速度惩罚压制了正向信号。
- **signal**: 此前加入的 hinge 横向/角速度惩罚在阈值处引入了不连续梯度，干扰了直立 gate 的学习，是 iter4/iter5 连续失败的潜在根因。
- **level**: Level 3
- **hypothesis**: 恢复至 iter3 的无阈值二次惩罚骨架，并提升 upright_bonus 权重，能稳定 gate 学习曲线，避免探索早期因惩罚阈值导致的梯度冲突，从而重新逼近并可能超越 1839 的历史最佳分。
- **risk**: 横向漂移可能暂时增大，但轻量二次惩罚足以在后期抑制；移除角速度惩罚可能让翻滚略有增加，但强化的 upright_bonus 和 gate 会间接约束姿态。若横向过大，后续可小幅提高 w_lat。
