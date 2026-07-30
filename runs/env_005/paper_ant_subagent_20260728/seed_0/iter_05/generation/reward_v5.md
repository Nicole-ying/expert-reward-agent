# 设计理由
## 信号覆盖审计
- 终止条件（摔倒、高度出界）有高度惩罚和直立奖励作为前兆，但缺少对**快速翻滚**（角速度）的直接惩罚。当 agent 因为步态不稳定导致身体剧烈翻滚时，当前奖励函数只能在身体已经倾斜到一定程度（通过 upright）或高度过低时才有反应，缺乏对翻滚过程的及时负反馈。
- 代理在 iter4 中学会了频繁摔倒的步态，一半 episode 提前终止（terminated=10/20），score_range [-1747, 13] 显示极差 episode 拉低了平均分。
- 组件表显示主信号（gated_forward + upright_bonus）已经足够，但缺乏稳定性信号来抑制导致摔倒的快速翻滚。

## 行为诊断
- **agent 在做什么？** 在 iter4 中，agent 采取了一种步态，导致约半数 episode 因身体高度出界或姿态崩溃而提前终止，其余 episode 也仅以较低的前进速度存活。这很可能是因为对侧向速度的惩罚形式改变后，agent 学会了更大幅度的侧向摆动，进而触发了不可控的翻滚。
- **干预目标**：提高姿态稳定性，通过在动作执行过程中直接惩罚过大的俯仰/滚转角速度，阻止身体进入不可恢复的翻转状态。
- **方向评估**：iter4 的改动导致崩溃，但与 iter3 方向一致（均在 gate+bonus 框架下小修），积累的失败次数未达 3 轮连续 ❌，可以继续在此骨架下改进。

## 选择干预层级：Level 2 — 结构变换（添加新组件）
- 当前组件中没有直接针对角速度的惩罚，但角速度是摔倒的直接前兆变量。添加一个 **hinge 形式的角速度稳定性惩罚**（基于 roll/pitch 角速度），能更早地提供纠正梯度。
- 仅添加一个新组件，保留其他组件不变。

## 新组件设计
- 使用 `obs[16]` (body_roll_velocity) 和 `obs[17]` (body_pitch_velocity) 的绝对值。
- 设置阈值 `ang_threshold = 1.0` rad/s，超过部分施加二次惩罚，避免对正常行走中的小幅摆动惩罚。
- 系数 `w_ang = 0.1`，使每步惩罚约 0–0.1，不超过主信号 per-step（~0.4）的 0.3 倍。
- 总惩罚负担（原 lateral_penalty + height_penalty + action_penalty 约 0.02/step + 新惩罚 ≤ 0.12/step）满足 ≤ 主信号 0.5 倍的要求。

## 预期效果
- 对快速翻滚的惩罚将让 agent 学会更平稳的步态，减少 episode 早期摔倒，从而恢复 iter3 的良好性能，并有希望超越。
- 风险：阈值或系数不当可能导致 agent 过于保守，前进速度降低。可通过后续迭代微调。

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

    # ---- angular velocity stability (roll/pitch hinge penalty) ----
    roll_vel  = obs[16]
    pitch_vel = obs[17]
    ang_threshold = 1.0
    w_ang = 0.1
    roll_excess  = max(0.0, abs(roll_vel)  - ang_threshold)
    pitch_excess = max(0.0, abs(pitch_vel) - ang_threshold)
    ang_penalty = -w_ang * (roll_excess**2 + pitch_excess**2)

    # ---- action smoothing (light energy regularizer) ----
    w_act = 0.005
    action_penalty = -w_act * sum(a**2 for a in action) / len(action)

    # ---- total reward ----
    total_reward = (gated_forward + upright_bonus +
                    height_penalty + lateral_penalty +
                    ang_penalty + action_penalty)

    components = {
        "gated_forward":      gated_forward,
        "upright_bonus":      upright_bonus,
        "height_penalty":     height_penalty,
        "lateral_penalty":    lateral_penalty,
        "ang_penalty":        ang_penalty,
        "action_penalty":     action_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 当前奖励缺少对翻滚角速度的直接惩罚，而这是高度的直接前兆，导致 agent 在不稳定步态下无法及时修正，最终频繁摔倒。
- **behavior**: agent 学到的步态导致半数 episode 提前摔倒，平均前进速度降低，整体分数暴跌至 -591。
- **signal**: 缺少稳定性惩罚（角速度过大），仅依赖高度和姿态的滞后信号。
- **level**: Level 2
- **hypothesis**: 添加 roll/pitch 角速度的 hinge 惩罚将抑制快速翻滚，减少摔倒并恢复前进能力。
- **risk**: 角速度惩罚过强可能导致 agent 动作过于僵硬，前进速度略有下降；后续可通过系数调整平衡。