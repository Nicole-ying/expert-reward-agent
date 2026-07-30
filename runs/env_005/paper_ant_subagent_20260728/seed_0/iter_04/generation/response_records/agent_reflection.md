# Response Record

# 设计理由

## 修改的组件：`lateral_penalty`

当前侧向惩罚是全步长、无界限的二次项 `-0.2 * v_y²`，平均每步约 -0.174，几乎与正向激励 `upright_bonus`（+0.19）完全抵消，没有对净得分形成正向贡献。

本次将全时二次惩罚 **变换为 hinge 形式**：只有当侧向速度绝对值超过 `0.5` 时才触发，并采用超出部分的二次惩罚，系数设为 `0.2`。这样设计是基于：
- **保留对极端侧向的抑制**：阈值 0.5 覆盖大部分正常小幅侧滑，允许必要的平衡调整，避免对一般步态产生不必要的惩罚。
- **大幅降低平均惩罚负担**：当前 v_y 均值约 0.92，若以超出 0.42 计算，新惩罚约 -0.035/步，比旧值下降约 80%，预计每回合多释放约 130~150 分，足以填补 160 分缺口。
- **校准合规**：修改后每步惩罚负担 ≤ 主信号（3.5/步）的 0.1x，远低于 0.5x 上限，且不会引发梯度崩溃。

其他组件保持不变：`gated_forward` 主信号、`upright_bonus`、`action_penalty` 以及近乎僵尸的 `height_penalty` 均维持原样，以最小改动试探提升。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- signal extraction ----
    body_z   = obs[0]
    quat_x   = obs[2]
    quat_y   = obs[3]
    v_x      = obs[13]  # forward velocity
    v_y      = obs[14]  # lateral velocity

    # ---- upright measure (continuous, 1.0 when perfectly upright, -1.0 inverted) ----
    up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---- upright gate (soft multiplication to forward progress) ----
    gate_lower = 0.3
    gate = (up_z - gate_lower) / (1.0 - gate_lower)
    gate = max(0.0, min(1.0, gate))

    # ---- gated forward progress ----
    w_fwd = 1.0
    gated_forward = w_fwd * v_x * gate

    # ---- upright bonus (continuous, provides direct posture gradient) ----
    w_up_bonus = 0.2
    upright_bonus = w_up_bonus * up_z

    # ---- body height safety (soft hinge quadratic, only near boundaries) ----
    z_low_safe  = 0.3
    z_high_safe = 0.8
    w_h = 0.5
    low_hinge  = max(0.0, z_low_safe - body_z)
    high_hinge = max(0.0, body_z - z_high_safe)
    height_penalty = -w_h * (low_hinge**2 + high_hinge**2)

    # ---- lateral stability (hinge penalty: only penalize when |v_y| > 0.5) ----
    w_lat = 0.2
    lat_threshold = 0.5
    lateral_excess = max(0.0, abs(v_y) - lat_threshold)
    lateral_penalty = -w_lat * (lateral_excess ** 2)

    # ---- action smoothing (light energy regularizer) ----
    w_act = 0.005
    action_penalty = -w_act * sum(a**2 for a in action) / len(action)

    # ---- total reward ----
    total_reward = (gated_forward + upright_bonus +
                    height_penalty + lateral_penalty + action_penalty)

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
- **audit**: height_penalty 近乎僵尸（active_rate 2.7%），lateral_penalty 与 upright_bonus 近乎完全对冲，无正向净值贡献；缺少提升净得分的有效杠杆。
- **behavior**: agent 已学会稳定前向行走（~981步，极少终止），但净得分被侧向惩罚明显拉低，存在用牺牲部分直立换取速度的痕迹。
- **signal**: 侧向惩罚过强且不分轻重，抑制了一般性侧滑调整，消耗了近半直立激励。
- **level**: Level 2
- **hypothesis**: 将 lateral_penalty 改为 hinge 形式并降低平均惩罚量，可释放约130~150分净收益，使 score 达到 2000 目标区间，同时仍对极端侧向保持约束。
- **risk**: 侧向惩罚减弱后，agent 可能增加侧向自由度，导致步态效率轻微下降或偏航增大，但 gated_forward 对前向速度的直接奖励将自动抑制这一趋势，风险可控。
