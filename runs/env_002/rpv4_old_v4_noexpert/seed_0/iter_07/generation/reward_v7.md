# 诊断报告

## evidence
- 全局记忆：`iter_1=296.08 → iter_2=237.50 → iter_3=192.43 → iter_4=298.56 → iter_5=254.16 → iter_6=299.43`（new best）。目标 300，当前差距 **0.57 分（0.19%）**。
- 本轮 `terminated=20/20, truncated=0/20, len=1322.95`，`score_range=[296.91, 301.78]`：无早终止，部分 seed 已超过 300，均值刚好卡在 300 下方。
- **iter_5→iter_6 改善 +45.27 分（>30），必须先回答改善来源**：iter_6 相对 iter_5 同时做了两件事——(a) forward 斜率 0.62 回退到 0.60，(b) 新增 `energy_penalty = -0.02 * Σ action²`。证据：`forward_progress ep_sum` 从 565.7 回升到 **593.6（历史最高）**；`energy_penalty ep_sum=-29.4`、active 100%、magnitude_share 4.5%，说明力矩正则确实在持续起作用；iter_6 的 checkpoint score 趋势是 `-96.8 → -97.4 → ... → 302.9` 的单调学习曲线，而 iter_5 的 checkpoint 趋势此前被记录为极不稳定。**结论：能量惩罚抑制了高频浪费力矩、使步态更平滑稳定，斜率回退恢复了 forward 梯度，二者共同促成改善。**
- 组件历史：`forward_progress = 504.9 → 555.1 → 333.0 → 587.3 → 565.7 → 593.6`。iter_6 拥有最高 forward ep_sum 和最高 score；iter_5 证明 slope 0.62 是错误方向。

## behavior_diagnosis
当前策略已经学到**稳定、快速、平滑的行走步态**：episode 长（len≈1323）、无早终止、forward_progress 历史最高、能量惩罚仅占 4.5%（辅助而非竞争目标）。这不是不健康行为，而是接近目标、只需最后一小步微调的健康状态。均值 299.43 只差 0.57 分，属于 seed 间方差内的临门一脚。

## signal_completeness
职责完备，无缺信号：
- `forward_progress`（主职责）：tanh 凹饱和，主导且正确（91% share 符合"主目标主导、辅助目标不混合"的要求）。
- `torso_stability` + `vertical_velocity_penalty`：防摔倒护栏，系数经 iter_1/iter_4 验证有效。
- `energy_penalty`：新增后有效平滑步态，是本次改善的关键，需保留。
- 不需要 contact/lidar 信号——当前步态没有单腿拖行或地形卡死证据。

## selected_level
**L1**。理由：目标差距 0.19% < 10%，已进入"接近目标优先 L1"区间；iter_6 是历史 best 且改善机制已理解；iter_5 在接近目标时动 forward 形态（斜率）导致崩盘的教训仍在。本轮只做单系数微调，不改任何数学形态。

## selected_intervention
只把 `forward_progress` 的幅度系数从 **2.0 → 2.05**（+2.5%），其余全部保持不变（tanh 斜率 0.60、能量惩罚 -0.02、稳定性/垂直项系数不变）。这是对主目标的均匀正向微调，不改变 tanh 的饱和形态，不会重新打开 iter_2 的高速刷分通道。

## falsifiable_hypothesis
当前运行速度下 `forward_progress` 约为 0.45/step，远低于 tanh 天花板 2.0，说明中等速度区间还有未被利用的奖励余量。把幅度均匀放大 2.5% 会提高中等速度区的 forward 梯度，促使 agent 走得更快一点，同时 tanh 仍在高速度饱和（不会重演 iter_2 的线性速度 exploit），能量惩罚继续压制高频抖动。预计均值从 299.43 推过 300。**若下一轮 score 下降或摔倒增多，则假设被证伪，应回退到 2.0 并改为微调能量惩罚系数。**

## expected_next_round
下一轮策略应保持相同的稳定平滑步态，但平均前进速度略快：`score ≥ 300`，`forward_progress ep_sum` 轻微上升（约 600–610），`energy_penalty / torso_stability` share 基本稳定，`terminated=20/20`、无早终止，`score_range` 整体上移。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    horizontal_velocity = next_obs[2]

    # L1 micro-tune: keep iter_6's proven concave tanh saturation and the
    # energy_penalty that drove the 254 -> 299 improvement. Only scale the
    # forward amplitude 2.0 -> 2.05 (+2.5%) to nudge the mean score from
    # 299.43 over the 300 target, without re-opening the high-speed farming
    # channel (tanh still saturates at high speed).
    if horizontal_velocity >= 0.0:
        v = min(horizontal_velocity, 4.0)
        # tanh(0.60 * v) = (e^(1.2*v) - 1) / (e^(1.2*v) + 1)
        e2v = 2.718281828 ** (1.2 * v)
        forward_progress = 2.05 * (e2v - 1.0) / (e2v + 1.0)
    else:
        # Asymmetric backward penalty: moving backward costs more than forward
        # gains, discouraging in-place oscillation.
        forward_progress = 1.5 * max(horizontal_velocity, -2.0)

    hull_angle = next_obs[0]
    hull_angular_velocity = next_obs[1]
    vertical_velocity = next_obs[3]

    # Stability and vertical terms held at iter_4/iter_6's proven coefficients.
    torso_stability = -0.15 * (hull_angle ** 2) - 0.05 * (hull_angular_velocity ** 2)
    vertical_velocity_penalty = -0.08 * (vertical_velocity ** 2)

    # Keep iter_6's torque-energy regularization unchanged: it was the key
    # improvement this round (smooth gait, suppressed high-frequency control).
    energy_penalty = -0.02 * (
        action[0] ** 2 + action[1] ** 2 + action[2] ** 2 + action[3] ** 2
    )

    total_reward = (
        forward_progress
        + torso_stability
        + vertical_velocity_penalty
        + energy_penalty
    )

    components = {
        "forward_progress": forward_progress,
        "torso_stability": torso_stability,
        "vertical_velocity_penalty": vertical_velocity_penalty,
        "energy_penalty": energy_penalty,
    }

    return float(total_reward), components
```