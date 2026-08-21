## 诊断报告

**evidence**
- 全局记忆：`iter_1=296.08`，`iter_2=237.50`，`iter_3=192.43`，`iter_4=298.56`（best），`iter_5=254.16`。目标 300，best 距目标仅 1.44 分（0.48%）。
- 组件历史：`forward_progress ep_sum = 504.9 → 555.1 → 333.0 → 587.3 → 565.7`。iter_4（tanh 0.6）同时拥有最高的 forward ep_sum（587.3）和最高 score（298.56）；iter_5 把斜率调到 0.62 后 forward ep_sum 反而降到 565.7，score 大跌到 254。
- iter_5 checkpoint 趋势极不稳定：score `-98.5 → -98.8 → -53.4 → 60.2 → 241.0 → 83.9 → -26.9 → 266.8 → 173.0 → 247.1`；forward ep_sum 在 `529 → 302 → 140 → 585 → 469 → 548` 之间反复震荡。
- `vertical_velocity_penalty` 五轮 ep_sum 恒为 -0.15~-0.47、magnitude_share ≤0.1%：是死组件，不产生任何学习信号。
- 环境卡片明确列出 `energy_regularization` / 动作平滑为条件职责，且“基础前进步态已出现后可加入作为效率优化项”。历史 5 轮 reward 均未使用 `action`，效率/平滑职责完全缺失。

**behavior_diagnosis**
- iter_5 的策略走得更久（len=1201 > iter_4 的 1043）但走得更慢：forward ep_sum 565.7 < iter_4 的 587.3。tanh 斜率从 0.6 提到 0.62 后，在中速工作点更接近饱和，反而**降低了对速度的正向梯度**，策略学会了“慢而久”而不是“快而稳”，score 从 298.56 跌到 254.16。
- 训练过程 score 剧烈震荡，说明当前奖励只有纯速度主信号 + 一个弱稳定项，没有任何对高能耗/高频力矩的抑制，策略在快慢步态之间来回切换，方差大。

**signal_completeness**
- 主职责 `forward_progress` 完备（tanh 凹饱和是历史最佳形态）；`torso_stability` 完备；`vertical_velocity_penalty` 死组件（无害但无用）。
- 缺失 `energy_regularization`（扭矩能耗/动作平滑）这一条件职责。当前主信号 96.5% 独占，没有效率压力，符合“基础步态已出现，可加入效率正则”的触发条件。

**selected_level**
- L2（新增一个组件）。唯一能回到 best 的纯 L1（斜率 0.62→0.60）已被判定为与 iter_4 完全重复；因此本轮采用最小幅度的**加法式新组件**作为实质性不同的干预，而非再次改动主信号数学形态。

**selected_intervention**
- 保留 iter_4 已被证明的主信号 `tanh(0.6*v)`（不碰 0.60 斜率），保留稳定项与垂直项系数。
- 新增 `energy_penalty = -0.02 * (a0² + a1² + a2² + a3²)`，把缺失的扭矩效率正则补上。系数取小，使其 magnitude_share 目标落在 ~1–3%，是辅助正则而非与“向前走”竞争的第二目标。

**falsifiable_hypothesis**
- 高能耗/高频力矩是 iter_5 训练震荡和步态不稳的可修正来源。加入小扭矩惩罚后，策略应减少浪费性力矩，用更平滑的步态维持与 iter_4 相当或更高的中速前进，从而 forward ep_sum 回升、score 向 300 靠拢、训练方差下降。
- 若下一轮 `energy_penalty` magnitude_share >10% 或 score 明显下滑，说明能量正则压制了有效动作，应减半系数。

**expected_next_round**
- 策略应保持中速前进（forward ep_sum ≈ 或 > iter_4 的 587），步态更平滑、扭矩更小，len 稳定在 1000–1200，score 接近或超过 298.56，且 checkpoint 之间的 score 震荡收窄。

## 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    horizontal_velocity = next_obs[2]

    # Forward progress: keep iter_4's proven concave tanh saturation.
    # iter_5's slope 0.62 was closer to saturation at the operating speed and
    # reduced the forward gradient (forward ep_sum 587 -> 565), slowing the gait.
    # Hold the slope at 0.60; this term is not the material change this round.
    if horizontal_velocity >= 0.0:
        v = min(horizontal_velocity, 4.0)
        # tanh(0.60 * v) = (e^(1.2*v) - 1) / (e^(1.2*v) + 1)
        e2v = 2.718281828 ** (1.2 * v)
        forward_progress = 2.0 * (e2v - 1.0) / (e2v + 1.0)
    else:
        # Asymmetric backward penalty: moving backward costs more than forward
        # gains, discouraging in-place oscillation.
        forward_progress = 1.5 * max(horizontal_velocity, -2.0)

    hull_angle = next_obs[0]
    hull_angular_velocity = next_obs[1]
    vertical_velocity = next_obs[3]

    # Stability and vertical terms held at iter_4's proven coefficients.
    torso_stability = -0.15 * (hull_angle ** 2) - 0.05 * (hull_angular_velocity ** 2)
    vertical_velocity_penalty = -0.08 * (vertical_velocity ** 2)

    # NEW component: mild torque-energy regularization.
    # No historical reward used `action`; the environment lists energy cost and
    # smooth hip/knee coordination as secondary objectives. A small quadratic
    # torque penalty suppresses wasted high-frequency control (the likely source
    # of iter_5's volatile score trend) without becoming a competing objective.
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