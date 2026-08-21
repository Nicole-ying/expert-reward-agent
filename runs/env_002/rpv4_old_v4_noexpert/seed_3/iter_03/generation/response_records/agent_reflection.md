# Response Record

## 诊断报告

**evidence**
- 历史得分：iter1 = 278.88 → iter2 = 285.19（+6.31），目标 300，当前差距 14.81（约 4.9%）。
- iter1→iter2 的唯一改动是把 `forward_reward` 系数从 1.0 提到 1.5（L1），`forward_reward` ep_sum 从 488.8 → 756.2，几乎按系数线性放大，说明速度信号有效传导。
- `balance_penalty` 从 -12.35 → -32.25（随速度提高而增大），但 magnitude_share 只有 4.1%，远没有支配奖励，说明稳定性没有成为主要瓶颈。
- checkpoint 趋势：前 60% 训练得分长期为负，70% 时从 -53.3 跳到 256.4，之后稳步上升到 284.3，`forward_reward` 同期从 130.7 跳到 726.2 并继续增长。这是典型的「后期学会走路并持续提速」曲线，不是 exploit。
- 组件历史：`forward_reward` active_rate 100%、`balance_penalty` active_rate 100%，没有死组件。

**behavior_diagnosis**
- 策略当前行为是健康的稳定前进步态：len=1110.5，20/20 均正常终止，score 范围紧凑（281.4–288.6），没有早期摔倒（early_terminal=0/20）。
- 不是纯平衡/站立不动（forward_reward 占主导且持续增长），而是「速度仍略低于目标」的前进型策略。
- 距离目标 4.9%，属于接近目标的状态，主要缺口是前进速度还不够快。

**signal_completeness**
- 必要职责完备：`forward_progress`（next_obs[2]）与 `balance_and_fall_prevention`（hull_angle、hull_angular_velocity、vertical_velocity）都已存在且都被学习到。
- 可选职责 `energy_efficiency` 暂不需要：当前得分接近目标，加入能量惩罚有减速风险，不引入。
- 显式 success/failure 信号不可用（info 为空），保持不依赖 terminal reward 的框架。

**selected_level**
- L1。理由：得分 285.19 与目标 300 差距 <10%，且上一轮已验证「提高 forward_reward 系数」这一语义方向是有效的（+6.3 分），应精准微调而不是改数学形态。

**selected_intervention**
- 目标组件：`forward_reward`。
- 具体修改：系数 `1.5 → 2.0`，其余（三个 balance 惩罚项的系数与形式）保持不变。
- 这是延续已证实的「把 speed/stability 权衡点向更快移动」的同一语义方向。

**falsifiable_hypothesis**
- 如果假设成立：更强的速度梯度会让策略在保持稳定（balance_penalty 份额仍低）的前提下走得更快，`forward_reward` ep_sum 进一步上升，score 向 292–296 逼近，len 不出现明显缩短（没有因为加速而大量摔倒）。
- 如果假设不成立：会出现 balance_penalty 份额骤升、early_terminal 增加或 score 回调，说明 2.0 已经越过稳定边界，需要回退或改结构。

**expected_next_round**
- 策略仍然是稳定前进步态，但平均前进速度更高；`balance_penalty` 略有上升但 magnitude_share 仍应 <10%；score 预计高于 285.19，向 300 靠近；terminated 仍 20/20 且不出现早期摔倒。

## 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # v3 reward: 延续 v1->v2 已验证的语义方向——加强前向速度驱动。
    # 证据：v2 将 forward_reward 系数从 1.0 提到 1.5 后，score 从 278.88 升到 285.19，
    # forward_reward ep_sum 从 488.8 升到 756.2，而 balance_penalty magnitude_share 仅 4.1%。
    # 说明稳定性已不是瓶颈，剩余缺口是前进速度略低。当前距目标 300 差约 4.9%，做 L1 微调：
    # 仅把 forward_reward 系数 1.5 -> 2.0，继续把 speed/stability 权衡点推向更快。
    # info 为空，不使用；无显式 success/failure 标志，不包含 terminal reward。
    _ = training_progress

    # 主学习信号：前向水平速度。obs[2] / next_obs[2] 约定为前向为正。
    forward_reward = 2.0 * next_obs[2]

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
