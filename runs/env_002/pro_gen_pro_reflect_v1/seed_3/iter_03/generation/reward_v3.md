# 设计理由

## 信号覆盖审计
- **终止 → 前兆**：环境中只有 `body_fallen_over` 是失败终止。当前代码已经通过 `hull_angle` → `health` 健康门控实现了前兆信号，但 `tilt_penalty` 是独立二次惩罚，与健康门控存在功能重叠。
- **目标 → 进度**：任务目标是最大化前进速度，`forward_progress = forward_vel * health` 直接对齐任务完成，设计合理。
- **效率信号**：动作维度为 4，当前有轻量 `action_cost`，per-step 约 0.018，合理。
- **僵尸组件**：所有组件 active_rate 均为 100%，无僵尸组件。
- **一句话结论**：当前奖励函数职责完备、无信号缺口，但 `tilt_penalty` 独立惩罚与健康门控功能重叠，且可能过度约束了高速移动时的微小倾斜，限制了 `forward_progress` 的提升空间。

## 行为诊断
- **agent 在做什么**：agent 已经掌握了稳定行走，20/20 terminated 均为到达终点（而非摔倒），len 从 1010 降至 885 表明移动更快、更早到达终点。行为健康。
- **干预目标**：核心目标是继续提升 `forward_progress`（当前 per-step 0.533，总分距目标差 15.86）。`tilt_penalty` 是当前约束 agent 倾斜过度的安全网，但 agent 已不摔倒，可以适当放松。
- **这个方向还值得继续吗**：累积记录显示 iter 1→2 是系数调整，分数从 217→284，预判均为 ✅。当前 skeleton 仍在上升通道，不值得重建。

## 选择干预层级：Level 1 — 尺度修复

`tilt_penalty` 是二次函数 `-5.0 * (hull_angle ** 2)`。当 `hull_angle` 在典型范围 0.1-0.2 弧度时，per-step 惩罚约 -0.05 到 -0.2，与 `forward_progress` per-step（约 0.53）相比负担不大。但在加速阶段，agent 可能需要更大的倾斜才能获得更高速度，此时二次惩罚快速增大，形成不必要的阻力。

**替代方案**：当前已有 `health` 门控直接压制倾斜下的 `forward_progress` 奖励，`tilt_penalty` 作为冗余安全网是合理的，但可以将系数从 `-5.0` 降低到 `-2.0`，减少其对高速倾斜的惩罚压力，同时保留安全网功能。

**系数校准**：per-step `tilt_penalty` 当前约 -0.0077。主信号 per-step 约 0.533。修改后，在 `hull_angle=0.15` 时 per-step 从 -0.1125 降至 -0.045，仍然远小于 0.3x 主信号，不会导致倾斜失控。

**备选更优方向（暂不采纳）**：另一种设计是将 `tilt_penalty` 改为 hinge `-2.0 * max(0, abs(hull_angle) - 0.1)`，让微小倾斜无惩罚。但由于当前 `health` 门控已经实现了这一点（倾斜较小时 health≈0.9-1.0），hinge 带来的额外收益有限。选择更保守的系数降低，风险更低。

## 修改内容
只改一个参数：`tilt_penalty` 系数从 `-5.0` 降至 `-2.0`。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Main learning signal: forward velocity, gated by posture health
    forward_vel = next_obs[2]                 # positive = moving forward
    hull_angle = next_obs[0]                  # radians, 0 = vertical
    health = 1.0 / (1.0 + 10.0 * (hull_angle ** 2))   # posture gate
    forward_progress = 1.0 * forward_vel * health

    # Stability: keep torso upright (relaxed safety net)
    tilt_penalty = -2.0 * (hull_angle ** 2)

    # Stability: suppress rotational jitter
    hull_angvel = next_obs[1]                 # rad/s
    angular_velocity_penalty = -1.0 * (hull_angvel ** 2)

    # Efficiency: light action cost
    avg_action_power = (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2) / 4.0
    action_cost = -0.05 * avg_action_power

    total_reward = forward_progress + tilt_penalty + angular_velocity_penalty + action_cost

    components = {
        'forward_progress': forward_progress,
        'tilt_penalty': tilt_penalty,
        'angular_velocity_penalty': angular_velocity_penalty,
        'action_cost': action_cost
    }
    return float(total_reward), components
```

---

# 诊断摘要
- **audit**: 职责完备无缺口，tilt_penalty 与 health gate 功能重叠但可作为安全网；降低其系数以减少对高速倾斜的过度惩罚。
- **behavior**: agent 已稳定行走并全部到达终点，速度还有提升空间。
- **signal**: forward_progress 是唯一主导信号（95.3%），tilt_penalty 在高速阶段可能过度约束。
- **level**: Level 1
- **hypothesis**: 降低 tilt_penalty 系数将减少在加速/倾斜阶段的惩罚负担，允许 agent 在保持健康门控的前提下追求更高速度，从而提升 forward_progress。
- **risk**: 倾斜角度略微增大（但 health gate 仍将压制极端倾斜对 forward_progress 的贡献）；不会导致摔倒。