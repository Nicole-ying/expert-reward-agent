## 诊断

**Agent 发生了什么？** Episode 长度 74.7，95% early terminal，得分 -220.92 → **agent 在 crash**。没有学到任何有效行为就死了。

**根因是什么？** 对比 Iter 3（最佳，score=169.76, len=922）和 Iter 4（当前，score=-220.92）：

| 指标 | Iter 3 | Iter 4 |
|---:|---|---|
| soft_landing_proxy mean | 0.294 | 0.000247 |
| soft_landing_proxy nonzero_rate | 正常（推测 >10%） | 0.29% |
| progress_reward mean | 0.043 | 0.136 |

上轮唯一改动是把 `soft_landing_proxy * 0.15`。这个 0.15 系数直接把 landing 信号压死了（nonzero_rate 从正常跌到 0.29%）。landing 信号是告诉 agent「到目标附近要减速、直立、着地」的唯一引导——没了它，agent 只剩 progress_reward 驱动它冲向原点但不会减速着陆，于是撞毁。

**为什么 Iter 3 的 ratio ~6.8 不是 bug？** Principle 1 明确说：终端/事件型奖励的 ratio 天然偏大，只要 nonzero_rate 正常且外部得分不差就不是问题。Iter 3 得分 169.76、长度 922，landing 信号的高 ratio 恰恰是它成功的原因——landing 需要足够强才能让 agent 学会在目标附近减速着陆，而不是被 progress 驱动冲过去。

## 行动：撤销 0.15 缩放

回到 Iter 3 的版本，移除 0.15 乘数。这是唯一改动，符合「每次只改一个信号」原则。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 位置
    cx, cy = float(obs[0]), float(obs[1])
    nx, ny = float(next_obs[0]), float(next_obs[1])

    d_curr = (cx ** 2 + cy ** 2) ** 0.5
    d_next = (nx ** 2 + ny ** 2) ** 0.5

    # 1. 主学习信号：逐步靠近目标（scale=8，符合 skeleton 推荐 5~20）
    progress_reward = (d_curr - d_next) * 8.0

    # 2. 轻量稳定约束
    vx, vy = float(next_obs[2]), float(next_obs[3])
    speed = (vx ** 2 + vy ** 2) ** 0.5
    angle = abs(float(next_obs[4]))
    angular_v = abs(float(next_obs[5]))

    raw_stability_penalty = (
        -0.005 * speed
        - 0.01 * angle
        - 0.005 * angular_v
    )

    # stage weighting: 早期忽略 stability，后期逐渐加入
    # t=0 → late_weight=0（无惩罚）；t=1 → late_weight=1（全惩罚）
    t = training_progress
    late_weight = max(0.0, min(1.0, 2.0 * t - 0.5))
    stability_penalty = raw_stability_penalty * late_weight

    # 3. 软着陆近似信号：连续乘积因子提供梯度
    # 每个因子 bounded 在 [0,1]，乘积天然稀疏但终端附近有强梯度
    proximity_factor = max(0.0, 1.0 - d_next / 0.3)       # 距离 < 0.3 时 >0
    speed_factor = max(0.0, 1.0 - speed / 0.5)            # 速度 < 0.5 时 >0
    angle_factor = max(0.0, 1.0 - angle / 0.3)            # 角度 < 0.3rad 时 >0
    contact_factor = (
        max(0.0, float(next_obs[6])) * max(0.0, float(next_obs[7]))
    )  # 两腿着地程度乘积

    soft_landing_proxy = proximity_factor * speed_factor * angle_factor * contact_factor

    # 总奖励
    total_reward = progress_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward,
    }
    return float(total_reward), components
```