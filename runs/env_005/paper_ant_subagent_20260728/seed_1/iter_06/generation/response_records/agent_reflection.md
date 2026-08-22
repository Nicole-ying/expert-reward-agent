# Response Record

# 设计理由
刚过去的第 5 轮给前进驱动增强了系数，同时加了横向、垂向的轻量惩罚，但最终前进速度反而从 iter 4 的 `≈0.64` 掉到了 `≈0.09`，说明新引入的惩罚干扰了探索，agent 陷入“慢速存活 + 身体倾斜（平均 `body_up_z≈‑0.37`）”的坏吸引子。  
证据：`upright_bonus` 触发率 100% 但 signed share 仅 2.1%，且均值就是负值——它实际上已经变成惩罚，而 0.2 的系数根本不足以纠正倾斜。身体倾斜又削弱了前进效率，使得整个生成奖励（+0.081/step）被环境自带的 -0.989/step 淹没。  

因此本轮**仅修改 upright 组件**：把“无方向性的线性奖励”换成 **hinge 惩罚**，只对倾斜（`body_up_z<0`）施加梯度，向上驱赶 body_up_z 到安全正值。这样在不大幅增加惩罚负担的前提下，给姿态纠正提供足够信号，让 agent 重新站起来并获得顺畅的前进速度。同时保留第 5 轮的 height_factor 和 forward 框架，不触碰其他组件，以验证单一改动的效果。  

系数校准：forward 当前 per‑step ≈ 0.182；`body_up_z≈‑0.37` 时新惩罚 ≈ `‑0.5×0.37 = ‑0.185`，与 forward 同量级。惩罚负担 < 0.5×主信号，未超过上限。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- Extract observations ----------
    body_z = obs[0]
    quat_x = obs[2]
    quat_y = obs[3]
    body_x_velocity = obs[13]
    body_y_velocity = obs[14]
    body_z_velocity = obs[15]

    # ---------- Derived signals ----------
    # Uprightness: 1.0 = perfectly upright, -1.0 = upside-down
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---------- Height gate (identical to iter4, proven effective) ----------
    height_low_danger = max(0.0, (0.30 - body_z) / 0.10)   # 0.30→0.20
    height_low_factor = max(0.0, 1.0 - height_low_danger)
    height_high_danger = max(0.0, (body_z - 0.90) / 0.10)  # 0.90→1.00
    height_high_factor = max(0.0, 1.0 - height_high_danger)
    height_factor = min(height_low_factor, height_high_factor)

    # ---------- Reward components ----------
    # Strong forward propulsion (inherited from iter5)
    forward_reward = 2.0 * body_x_velocity * height_factor

    # Hinge upright penalty — activates only when tilted (body_up_z < 0)
    # Replaces weak upright_bonus with a corrective gradient
    upright_penalty = -0.5 * max(0.0, 0.0 - body_up_z)

    # Light hinge penalties (unchanged from iter5)
    lateral_deviation = abs(body_y_velocity)
    lateral_penalty = -0.1 * max(0.0, lateral_deviation - 0.3)

    vertical_deviation = abs(body_z_velocity)
    vertical_penalty = -0.1 * max(0.0, vertical_deviation - 0.5)

    # ---------- Total reward ----------
    total_reward = forward_reward + upright_penalty + lateral_penalty + vertical_penalty

    components = {
        "forward": forward_reward,
        "upright_penalty": upright_penalty,
        "lateral_penalty": lateral_penalty,
        "vertical_penalty": vertical_penalty,
        "height_factor": height_factor,
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 终止前兆 ✓（高度 gate 充足），目标进度 ✓（forward 有梯度），缺能量效率信号（暂缓）；当前奖励被 env penalty 淹没，根因是 upright 信号太弱导致身体倾斜 → 前进瘫痪。
- **behavior**: agent 存活 713 步但几乎不移动（vx≈0.09），处于“倾斜慢速存活”的坏平衡点。
- **signal**: upright 组件完全失效——名义 bonus 已是事实惩罚，幅度不足以纠正姿态。
- **level**: Level 2 — 结构变换（upright 从 无界弱奖励 → hinge 倾斜惩罚）
- **hypothesis**: 给倾斜施加强梯度会迫使 agent 快速恢复直立，从而使 forward 奖励恢复 iter 4 水平，分数回正。
- **risk**: 若 agent 一开始直立困难，hinge 惩罚会让早期总分更负，可能减慢学习；但惩罚量级与 forward 相当，不会形成惩罚占优的死亡螺旋。
