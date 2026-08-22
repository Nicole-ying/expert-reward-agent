## 诊断报告

**evidence**
- 全局记忆：`iter_1 score=296.08`（best），`iter_2=237.50`，`iter_3=192.43`，`iter_4=298.56`（新 best）。目标 300，本轮差距 1.44 分（0.48%），已进入接近目标区间。
- 本轮 `terminated=20/20`、`truncated=0/20`、`len=1043.15`、`score_range=[294.56, 301.71]`：策略稳定行走，无早终止，部分 seed 已超过 300。
- 组件历史：`forward_progress ep_sum = 504.9 → 555.1 → 333.0 → 587.3`；`torso_stability ep_sum = -23.7 → -20.5 → -4.9 → -30.6`。
- checkpoint 趋势：`forward_progress 11.4 → 547.2` 全程单调增长，无骤升骤降；score 从 cp70 起 `130.4 → 267.5` 持续上升，最终 eval 达 298.56，说明 tanh 形式没有被 exploit。

**iter_3→iter_4 改善原因（必须先回答）**
- 改善幅度 +106.1 分（192.43 → 298.56）。促成改善的是 **把线性速度 proxy（`0.8*min(v,2)`）换成凹形 tanh 饱和奖励（`2*tanh(0.6v)`）**，同时把稳定性系数从 iter_3 的过强值 `-0.20/-0.07` 恢复到 iter_1 的已验证值 `-0.15/-0.05`。
- 证据：iter_2 曾把线性 forward 从 `1.0/1.5` 加到 `1.2/1.8`，forward ep_sum 升到 555，但外部 score 反而跌到 237、len 缩短——典型 proxy 被高速 exploit；iter_3 又把线性系数压到 `0.8/1.2`，forward 塌缩到 333、score 192，过度矫正。iter_4 的 tanh 在中等速度附近保留了 iter_1 的奖励强度（v=1→1.07，v=2→1.67），但对高速收益递减（上界 2.0），因此 forward ep_sum 升到 587 的同时外部 score 回到并超过 iter_1，checkpoint 全程无 exploit 回落。

**behavior_diagnosis**
- 策略已学会快速、稳定的向前行走：20/20 正常终止、len 1043、score 298.56 接近目标。`forward_progress` 占 95% 是主目标主导的正常现象，不是 exploit（tanh 有界且 checkpoint 趋势无回落）。`torso_stability=-30.6`（4.9%）说明步态比 iter_1 更激进但仍在可接受范围，没有摔倒早终止。
- 当前不是不健康行为，只是均值差 1.44 分过线。

**signal_completeness**
- 必要职责完备：`forward_progress`（主目标前进）、`torso_stability`（防摔倒）、`vertical_velocity_penalty`（抑制躯干起伏）都在。不需要新增能量/平滑项（环境卡明确它们是辅助目标，不应与主目标混成多目标）。当前不缺信号，缺的只是主信号强度的一点点微调。

**selected_level**
- **L1**。理由：得分距目标仅 0.48%，远小于 10% 阈值；且历史经验（iter_2）显示靠近目标时把 forward 加大 20% 会越改越差。本轮只做系数级微调，不改数学形态。

**selected_intervention**
- 目标组件：`forward_progress` 的正向分支。
- 具体修改：tanh 斜率 `0.6 → 0.62`，饱和上界 `2.0` 保持不变。
- 效果：在中等速度区间把 forward 奖励提高约 2–3%，但不抬高高速 exploit 天花板（上界仍为 2.0），在保留 iter_4 已验证凹形结构的前提下，给策略一个“再快一点点”的微弱激励。

**falsifiable_hypothesis**
- 因为外部 score 与前进速度同向（iter_4 比 iter_1 更快、len 更短、分数更高），小幅提高中等速度的奖励会让策略把巡航速度略微提高，从而把均值从 298.56 推过 300；由于 tanh 上界未变，不会重新打开 iter_2 的线性高速 farming 通道。
- 若下一轮 score 下降或 torso_stability 量级显著增大，则说明该 nudge 仍偏大，需要回退到 `0.6` 或改用更保守方向。

**expected_next_round**
- 策略仍是快速稳定行走，`len` 可能略微缩短或持平，`forward_progress` ep_sum 略升，`torso_stability` 基本持平，外部 score 均值预计越过 300。

## 完整代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    horizontal_velocity = next_obs[2]

    # L1 micro-tune: keep iter_4's proven concave tanh saturation, raise the
    # moderate-speed slope 0.6 -> 0.62 while keeping the 2.0 ceiling unchanged.
    # This nudges forward reward up ~2-3% in the mid-speed range (the operating
    # point) to push the mean score from 298.56 over 300, without re-opening the
    # high-speed farming channel that crashed iter_2.
    if horizontal_velocity >= 0.0:
        v = min(horizontal_velocity, 4.0)
        # tanh(0.62 * v) = (e^(1.24*v) - 1) / (e^(1.24*v) + 1)
        e2v = 2.718281828 ** (1.24 * v)
        forward_progress = 2.0 * (e2v - 1.0) / (e2v + 1.0)
    else:
        # Keep iter_1's proven asymmetric backward penalty: moving backward is
        # punished harder than moving forward is rewarded, discouraging oscillation.
        forward_progress = 1.5 * max(horizontal_velocity, -2.0)

    hull_angle = next_obs[0]
    hull_angular_velocity = next_obs[1]
    vertical_velocity = next_obs[3]

    # Hold stability and vertical terms at iter_1/iter_4's proven coefficients:
    # they were not the exploit source, and changing them risks trading away
    # forward progress (as seen in iter_3) or destabilizing the gait.
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