## 诊断报告

**evidence**
- `iter_1` score=278.88 / target=300，差距 21.1 分（≈7%），处于“接近目标”区间；len=1077.7，terminated=20/20，truncated=0/20，且 `early_terminal(<150 且 score<-50)` 为 0/20，说明没有早期摔倒的灾难样本。
- 组件组成：`forward_reward` ep_sum=488.8（magnitude_share 97.5%），`balance_penalty` ep_sum=-12.35（2.5%）。平均前向速度 ≈ 488.8/1077.7 ≈ 0.45 m/s，躯干倾斜惩罚很小 → 策略已经稳定向前走。
- checkpoint 趋势：forward_reward 从 cp10 的 12.9 单调增长到 cp100 的 479.2；balance_penalty 全程小且最终收敛到 -10.2。score 从 -94.7 波动上升至 271.8（cp100）→ 278.9（final）。趋势健康，主信号在持续引导前进。
- `get_component_history` 确认：forward_reward 是唯一主导组件，但它是任务主目标，不是被 exploit 的 proxy。

**behavior_diagnosis**
策略已经学会“稳定向前行走”：躯干接近竖直（倾斜惩罚极小），没有站立不动（forward_reward 很高），没有 proxy 徘徊，也没有摔倒级联。行为不健康的部分不是稳定性，而是**前进速度还不够快**——这是距离目标 300 的剩余缺口。len=1077.7 且全部 terminated，说明策略在固定地形上行走，提升 score 的主要杠杆是单位时间前向速度。

**signal_completeness**
- 主职责 forward_progress：已具备（线性速度奖励），方向正确。
- 健康约束 balance：已具备且当前处于被满足状态（惩罚很小是好事，不是失衡）。
- 显式 terminal success/failure：info 为空，不可用，不补。
- 能量效率：条件职责，当前不是瓶颈，加入会反向减速，暂不引入。
信号职责基本完备，缺的不是新信号，而是主信号强度。

**selected_level**
**L1**。理由：差距 <10%（≈7%），且当前行为方向正确、组件职责齐全、训练趋势健康，只需强化主目标激励，不需要改数学形态或换框架。

**selected_intervention**
目标组件：`forward_reward`。具体修改：系数 1.0 → 1.5，`balance_penalty` 结构不变。这会把 speed/stability 权衡点向“更快”推移，而平衡惩罚仍在近竖直工作点提供梯度防止摔倒。

**falsifiable_hypothesis**
提高前向速度奖励权重后，agent 会被更强地激励提升水平速度，平均 `next_obs[2]` 上升 → 外部前进得分从 ~279 向 300 靠拢；同时 balance_penalty 仍保持梯度，预计不会出现摔倒显著增多。若下一轮 forward_reward 的 ep_sum 显著上升但外部 score 不升反降，则说明出现了“高速但不稳定/无效步态”，需要回到平衡侧收紧。

**expected_next_round**
策略步态更快，平均水平速度上升，forward_reward ep_sum 明显提高，score 更接近 300；balance_penalty 可能小幅上升（速度提高带来的倾斜/角速度增加），但不应量级爆发。

## 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # v2 reward: 强化前向速度驱动，保持躯干/身体稳定性正则。
    # 诊断：v1 已稳定行走（balance_penalty 仅 2.5% magnitude_share），
    # 剩余缺口是前进速度不够快。因此 L1 提高 forward_reward 系数，
    # 让 speed/stability 权衡点向"更快"移动。
    # info 为空，不使用；无显式 success/failure 标志，不包含 terminal reward。
    _ = training_progress

    # 主学习信号：前向水平速度。obs[2] / next_obs[2] 约定为前向为正。
    forward_reward = 1.5 * next_obs[2]

    # 稳定性/防摔倒约束：惩罚躯干倾斜、剧烈角速度和过度垂直运动。
    tilt_penalty = 1.0 * next_obs[0] ** 2
    angular_velocity_penalty = 0.2 * next_obs[1] ** 2
    vertical_velocity_penalty = 0.1 * next_obs[3] ** 2

    balance_penalty = (
        tilt_penalty
        + angular_velocity_penalty
        + vertical_velocity_penalty
    )

    total_reward = forward_reward - balance_penalty

    components = {
        "forward_reward": forward_reward,
        "balance_penalty": -balance_penalty,
    }

    return float(total_reward), components
```