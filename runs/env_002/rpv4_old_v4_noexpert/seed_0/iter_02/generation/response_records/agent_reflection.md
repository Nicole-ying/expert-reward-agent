# Response Record

## 诊断报告

**evidence**
- iter_1 最终 `score=296.08`，目标 300，差距仅 3.92 分（1.3%），属于接近目标区间。
- `len=1173.6`，`terminated=20/20`，`truncated=0/20`：策略能长距离稳定行走，无早终止/截断。
- 组件：`forward_progress` ep_sum=504.92、magnitude_share=95.5%、active=100%；`torso_stability`=-23.70（4.5%）；`vertical_velocity_penalty`=-0.27（0.1%）。
- checkpoint 趋势：score 从 cp10 的 -95 → cp70 的 256.5 → cp90 的 298.8 → cp100 的 296.0；`forward_progress` 从 cp60 起主导（213.8→480.6→505.3→504.7），是 score 上升的主要驱动。
- cp90（298.8）→ cp100（296.0）的微小回落伴随 `forward_progress` 微降（505.3→504.7），而非稳定性变差（torso 惩罚反而从 -27.4 收窄到 -23.6），说明外部得分对前进量/前进速度高度敏感。

**behavior_diagnosis**
- 策略已经学会稳定向前行走：步态长、全程向前、躯干较稳、无死亡/截断。这是健康行为，不是 proxy 徘徊或 exploit。
- `forward_progress` 占 95.5% 是合理的——它是主目标，且 agent 已把躯干稳定等惩罚项压低，因此 share 高属于“任务已学好”而非“单信号劫持”。
- 最后一小段 checkpoint 显示前进量几乎平台化，最终 score 略低于 cp90 峰值，说明还差最后一小步“更快/更远”的推进力。

**signal_completeness**
- 必备职责完备：`forward_progress`（主目标）✓、`torso_stability`（防摔倒）✓。
- `vertical_velocity_penalty` 存在但贡献极小（0.1%），无害，可保留。
- 能量/动作平滑是辅助目标，当前阶段不应与“向前行走”主目标混合成多目标问题，因此本轮不引入。
- 缺少显式 success/failure 标志，无法做稀疏终点奖励；这由连续 forward 信号替代，方向正确。

**selected_level**
- L1。当前 296.08 距目标 300 仅 1.3%，且上一轮方向明显正确（forward 主驱动、策略已稳定行走），不应改数学形态或换框架，只做系数微调。

**selected_intervention**
- 目标组件：`forward_progress`。
- 具体修改：正向前进系数 `1.0 → 1.2`；后退惩罚系数 `1.5 → 1.8`，保持 1.5 倍的前/后非对称，继续抑制后退。其余组件不动。

**falsifiable_hypothesis**
- 摔倒会立即终止 episode 并切断后续所有 forward 奖励，因此“每步多给一点前进奖励”会提高 agent 对持续快速向前的估值，而不会显著增加摔倒风险（摔倒本身已是灾难性损失）。
- 预计该微调会小幅抬高平均前进速度 / `forward_progress`，把外部 score 从 ~296 推向 ≥300。

**expected_next_round**
- 策略保持稳定步态，`forward_progress` 小幅上升，`torso_stability` 与 `vertical_velocity_penalty` 量级基本不变，score 预计接近或超过 300，且不出现原地摆动或高频抖动等 proxy 行为。

## 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    horizontal_velocity = next_obs[2]

    # Forward progress: encourage positive horizontal velocity, penalize backward motion.
    # L1 micro-adjustment: strengthen the primary forward signal slightly.
    if horizontal_velocity >= 0.0:
        forward_progress = 1.2 * min(horizontal_velocity, 2.0)
    else:
        forward_progress = 1.8 * max(horizontal_velocity, -2.0)

    hull_angle = next_obs[0]
    hull_angular_velocity = next_obs[1]
    vertical_velocity = next_obs[3]

    torso_stability = -0.15 * (hull_angle ** 2) - 0.05 * (hull_angular_velocity ** 2)
    vertical_velocity_penalty = -0.08 * (vertical_velocity ** 2)

    total_reward = forward_progress + torso_stability + vertical_velocity_penalty

    components = {
        "forward_progress": forward_progress,
        "torso_stability": torso_stability,
        "vertical_velocity_penalty": vertical_velocity_penalty,
    }

    return float(total_reward), components
```
