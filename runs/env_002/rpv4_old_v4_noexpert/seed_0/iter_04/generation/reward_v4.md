## 诊断报告

**evidence**
- 全局记忆：`iter_1 score=296.08`（best，距目标 300 仅 1.3%），`iter_2 score=237.50`，`iter_3 score=192.43`；三轮全部 `terminated=20/20`，`truncated=0/20`，无早终止（`<150 steps` 为 0）。
- `forward_progress` 组件历史：`ep_sum_mean = 504.9 → 555.1 → 333.0`。关键反例是 iter_2：训练奖励里 forward proxy 明明升高（505→555），外部 score 却从 296 跌到 237，且 len 从 1173.6 缩短到 1050.8 —— 典型 **proxy 提高但外部任务不升**（线性速度奖励被 exploit）。
- iter_3 把 forward 从 1.2/1.8 压到 0.8/1.2，同时把 stability 提到 -0.20/-0.07，结果 forward ep_sum 崩到 333、外部 score 跌到 192 —— 过度保守、矫枉过正。
- 三轮 `forward_progress` 的代码形态完全一致，都是 `coef * min/max(v, ±2.0)` 的线性裁剪，只动了系数（L1）；两个方向（加强、减弱）都已试过且都变差。
- `torso_stability` 与 `vertical_velocity_penalty` 不是 exploit 源：iter_1 的 stability=-23.7（4.5%）是健康占比，vertical 仅 0.1% 且从未改变，属可忽略项。

**behavior_diagnosis**
- iter_1 学到的是中速稳定步态（len≈1173，外部 296），是当前唯一接近目标的策略。
- iter_2 在更强的线性速度奖励下学到了"更快但更早摔"的激进步态：forward proxy 上升、len 缩短、外部分下降，说明线性无衰减的速度奖励在高速度区间被 exploit。
- iter_3 在削弱速度 + 加强稳定性后学到"慢而稳但不再前进"的保守步态：len 拉长到 1122，但 forward ep_sum 和外部分同步塌缩。
- 结论：问题不在 stability，而在 **forward_progress 的线性形态**——它给高速度提供与速度成正比的无衰减边际收益，导致只能在两个极端之间跳，无法稳定落在 iter_1 的中速甜点。

**signal_completeness**
- 主职责完备：forward_progress（前进）+ torso_stability（防摔）都在，且都有 100% active_rate。
- 缺的不是新信号，而是 forward 信号的**边际收益形状**：线性奖励让"再快一点"始终值得，直至摔倒；缺少"中速就够、极速收益递减"的饱和形态。
- vertical_velocity_penalty 量级 0.1%，实际是死项，但无害，暂不处理（保持单语义修改）。

**selected_level**：L2
- 理由：best=296.08 距目标仅 1.3%，本应优先 L1，但 L1 的两个方向（iter_2 加强、iter_3 减弱）均已尝试且都显著变差，且"恢复 iter_1 系数"的草稿被判定为语义重复。这说明继续在同一线性形态上调系数没有搜索价值。需要改变 forward_progress 的**数学形态**，而不是继续调系数。

**selected_intervention**
- 目标组件：`forward_progress`。
- 具体修改：把正速度分支从 `1.0 * min(v, 2.0)`（线性裁剪）改为 **tanh 饱和**：`2.0 * tanh(0.6 * v)`，负速度分支保留 iter_1 已验证的线性非对称惩罚 `1.5 * max(v, -2.0)`。stability 与 vertical 保持 iter_1 的已验证系数（-0.15/-0.05、-0.08），不引入额外多目标项。

**falsifiable_hypothesis**
- 线性形态下，高速度的边际收益恒定（甚至在有 cap 前恒定），agent 会为 proxy 堆速度直至摔倒（iter_2）。改成 tanh 后，中速区（0.5–1.0 m/s）奖励与 iter_1 基本持平甚至略高（鼓励持续前进、防止原地停），但高速度区（>1.5 m/s）边际收益递减，proxy 不再能被"极限速度"刷高。
- 预计新策略的 `forward_progress` ep_sum 落在 iter_1(505) 与 iter_3(333) 之间的健康区间，外部 score 从 192 回升向 250–296，len 保持 1100+，不出现 iter_2 的早摔缩短。
- 若下一轮 forward ep_sum 仍被刷到 550+ 且外部分下降，说明饱和点/斜率仍偏高，需要继续下调 tanh 斜率（L1 微调）；若 forward ep_sum 塌到 <350 且外部分不升，说明饱和过度惩罚了前进，需要上调斜率。

**expected_next_round**
- 策略学习到"持续中速前进、不追求极限速度"的稳定步态；训练 reward 中 forward_progress 占比从 iter_3 的 98.5% 略降回 iter_1 的 ~95%，torso_stability 负分回到 -20 左右的健康水平，外部 score 明显高于 iter_2/iter_3。

## 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    horizontal_velocity = next_obs[2]

    # L2: replace the linear speed reward (coef * min(v, 2.0)) with a concave
    # tanh saturation. Evidence: iter_2 raised the linear forward proxy from
    # 505 -> 555 ep_sum but the external score DROPPED 296 -> 237 with a shorter
    # episode (1173 -> 1050): the agent exploited raw high speed until falling.
    # iter_3's linear 0.8/1.2 over-corrected it (forward ep_sum 333, score 192).
    # tanh keeps moderate-speed reward close to the proven iter_1 balance while
    # giving diminishing returns at high speed, so the proxy can no longer be farmed.
    if horizontal_velocity >= 0.0:
        v = min(horizontal_velocity, 4.0)
        # tanh(0.6 * v) = (e^(1.2*v) - 1) / (e^(1.2*v) + 1)
        e2v = 2.718281828 ** (1.2 * v)
        forward_progress = 2.0 * (e2v - 1.0) / (e2v + 1.0)
    else:
        # Keep iter_1's proven asymmetric backward penalty: moving backward is
        # punished harder than moving forward is rewarded, discouraging oscillation.
        forward_progress = 1.5 * max(horizontal_velocity, -2.0)

    hull_angle = next_obs[0]
    hull_angular_velocity = next_obs[1]
    vertical_velocity = next_obs[3]

    # Hold stability and vertical terms at iter_1's proven coefficients: they were
    # not the exploit source, and iter_3 showed that over-strengthening stability
    # trades away forward progress.
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