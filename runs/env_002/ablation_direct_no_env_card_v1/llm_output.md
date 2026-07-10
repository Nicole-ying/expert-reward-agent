# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── 主学习信号：向前速度 ──
    forward_velocity = next_obs[2]
    w_forward = 1.0
    forward_reward = w_forward * forward_velocity

    # ── 姿态约束：保持身体直立 (hinge 允许小幅摆动) ──
    hull_angle = next_obs[0]
    w_angle = 0.5
    angle_threshold = 0.2   # 允许 ±0.2 rad 的自由摆动区间
    posture_penalty = -w_angle * max(0.0, abs(hull_angle) - angle_threshold)

    # ── 能耗代价：动作平方和 ──
    w_energy = 0.01
    action_penalty = -w_energy * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)

    # 总奖励
    total_reward = forward_reward + posture_penalty + action_penalty

    components = {
        "forward_reward": forward_reward,
        "posture_penalty": posture_penalty,
        "action_penalty": action_penalty
    }
    return float(total_reward), components
```

# 设计说明

- **任务类型**：2D bipedal locomotion（双足步行），要求走得远、走得快，同时不倒下并节约能量。
- **选取的信号与职责**：
  - **forward_reward**（主学习信号）：使用 `next_obs[2]`（horizontal velocity），直接奖励向前速度，驱动agent产生持续前进的步态。
  - **posture_penalty**（安全约束）：使用 `next_obs[0]`（hull angle），对主身体倾角超出安全区间的部分施以hinge惩罚，既抑制倾倒风险，又允许行走时必要的身体摆动。
  - **action_penalty**（效率代价）：使用动作的平方和，轻微抑制过大扭矩，鼓励节能。
- **数学形式选择依据**：
  - 前进奖励采用线性正信号（`w * signal`），因为速度本身连续且无饱和风险。
  - 姿态约束选用hinge形式（`-w * max(0, |angle|-threshold)`），避免在正常摆动范围内持续惩罚，仅在大倾角时施加梯度，符合专家 Schema 中对安全约束的建议。
  - 动作代价为二次项，权重极小（0.01），不会压制主学习信号。
- **排除的职责及原因**：
  - **terminal_success_reward / terminal_failure_penalty**：v1阶段避免稀疏终端奖励，专注于稠密驱动。
  - **vertical velocity 约束**：仅靠前进奖励不足以稳定，但姿态约束已间接限制翻滚，避免组件过多。
  - **leg contact 奖励**：不需硬性引导踩地，行走中自然会形成接触，过早加入可能造成依赖。
  - **能量效率用更复杂形式**：后续可根据训练曲线调整权重或增加门控。
- **应观察的failure modes**：
  - 若agent前进缓慢、步态过于保守，可能需提升`w_forward`或放宽`angle_threshold`。
  - 若agent频繁剧烈倾倒但角度惩罚active_rate高，说明hinge区间太窄或权重不足，可增大`w_angle`或收紧阈值。
  - 若动作代价导致动作幅度过小、无法前进，应减小`w_energy`。