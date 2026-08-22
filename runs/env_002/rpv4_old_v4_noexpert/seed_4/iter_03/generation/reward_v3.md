## 1. 诊断报告

**evidence**
- `read_memory()`：iter_1（无能量项）score=278.99, len=1275.6；iter_2（加 `-0.03*action_sq`）score=274.34, len=1427.05。加入能量惩罚后分数下降 4.65，回合反而变长。
- 每步前向速度反推：
  - iter_1：`forward_progress/ep_len = 697.414/1275.6 ≈ 0.5466`，对应 `v ≈ 0.376`。
  - iter_2：`720.870/1427.05 ≈ 0.5052`，对应 `v ≈ 0.338`。
  - iter_2 总前进量 ≈ `0.338×1427 ≈ 482`，iter_1 ≈ `0.376×1275 ≈ 479`——**总位移几乎相同，但 iter_2 走得更慢、用了更多步**。
- `read_training_feedback(2)`：`score_range=[54.5, 295.2]`，最佳单集 295.2（接近目标 300），均值被少数低分集拖低；说明策略已经能走完大半，瓶颈主要在速度/效率而非基本平衡。
- `read_checkpoint_trend(2)`：训练曲线比 iter_1 平滑（能量项起到了正则化作用），但 cp90→cp100 分数从 289.6 掉到 269.9，说明后期能量/稳定性惩罚在压制前进。
- 组件量级：iter_2 中 `forward_progress` magnitude_share=88.8%，`energy`=5.9%，`upright`=5.3%。前进仍是唯一主驱动，未被 exploit，但两个次级惩罚合计约 11% 已足以把步态拖慢。

**behavior_diagnosis**
- 策略已学会稳定行走并接近地形尽头（len 长、terminated 全部触发），不是原地抖动或频繁摔倒。
- 加入能量惩罚后，策略选择「更慢但更省力」的步态：总位移没变，但每步速度从 0.376 降到 0.338，步数从 1275 增到 1427，外部分数反而下降。
- 这是次目标（能量）压制了主目标（前进速度）的不健康倾向：能量项不该取代前进。

**signal_completeness**
- 主信号 `forward_progress`（前进速度）齐备且有效。
- `upright_stability` 齐备；iter_1 时 ep_sum 仅 -6.7，说明稳定性有充足余量，不是当前瓶颈。
- 能量信号是 conditional role，但当前权重偏大、且施加方式（无条件每步惩罚）在早期/全期都压制力矩输出，拖慢步态。
- 不缺必要信号，缺的是主/次目标的权重再平衡。

**selected_level**
- **L1**。当前得分 274.34 距目标 300 差距约 8.6%（<10%），且 best=295.2 已接近目标；之前经验显示接近目标时做 L2 容易越改越差。问题方向明确（能量项过强压制速度），只需调系数。

**selected_intervention**
- 目标组件：`energy_efficiency_penalty` 与 `forward_progress`（同一语义：把奖励平衡重新拉回「前进为主、能量为辅」）。
- 具体修改：
  - `energy_efficiency_penalty` 系数 `0.03 → 0.01`（弱化压制速度的次目标，保留轻微正则化以维持训练平滑）。
  - `forward_progress` 增益 `2.0 → 2.5`（增强主目标，鼓励更快步态，向 300 靠拢）。
- 数学形态不变，仅调系数。

**falsifiable_hypothesis**
- iter_2 的 `-0.03*action_sq` 让策略为省力而放慢脚步（v 0.376→0.338，len 1275→1427），总位移未增而分数下降。
- 若把能量惩罚降到 1/3，同时把前向增益提高 25%，则每步速度应回升并超过 iter_1 的 0.376，回合更短、外部分数更高；若下一轮每步 forward_progress 未升、len 未降或分数未改善，则说明瓶颈不在速度/能量权重，需转向稳定性或终止信号方向。

**expected_next_round**
- `forward_progress` 每步均值上升（对应 v > 0.376），`energy_efficiency_penalty` ep_sum 从 -48 降到约 -15 上下，`len` 缩短到 1400 以下，score 超过 274 并有望超过 iter_1 的 279。

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前向速度，使用光滑有界形式。
    # 增益从 2.0 提升到 2.5，强化主目标，鼓励更快步态。
    horizontal_velocity = next_obs[2]
    forward_progress = 2.5 * horizontal_velocity / (1.0 + abs(horizontal_velocity))

    # 稳定性约束：惩罚躯干倾角与躯干角速度（保持不变，稳定性不是当前瓶颈）
    torso_angle = next_obs[0]
    torso_angle_sq = torso_angle * torso_angle
    angle_penalty = 1.5 * torso_angle_sq / (1.0 + torso_angle_sq)

    torso_angular_velocity = next_obs[1]
    torso_angular_velocity_sq = torso_angular_velocity * torso_angular_velocity
    angular_velocity_penalty = 0.5 * torso_angular_velocity_sq / (1.0 + torso_angular_velocity_sq)

    upright_stability = -(angle_penalty + angular_velocity_penalty)

    # 能量效率（次目标）：从 0.03 降到 0.01。
    # 之前 0.03 的惩罚把策略拖慢（每步速度 0.376 -> 0.338），次目标压过了主目标。
    # 保留轻微惩罚以维持训练平滑，但不让它取代前进速度。
    action_sq = sum(a * a for a in action)
    energy_efficiency_penalty = -0.01 * action_sq

    total_reward = forward_progress + upright_stability + energy_efficiency_penalty

    components = {
        "forward_progress": forward_progress,
        "upright_stability": upright_stability,
        "energy_efficiency_penalty": energy_efficiency_penalty,
    }

    return float(total_reward), components
```