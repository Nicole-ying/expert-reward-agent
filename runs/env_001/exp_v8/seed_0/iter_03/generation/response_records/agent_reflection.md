# Response Record

## 诊断

看完反馈后，我的判断如下：

**1. Agent 发生了什么？**
Episode 长度 68 步，100% early terminal，score=-108。Agent 在快速 crash，没有学到任何有效行为。`original_env_reward` mean=-1.575（燃料+crash），而我们的 `total_reward` 只有 0.013——环境本身的惩罚完全压倒了奖励信号。

**2. 哪个组件是主要原因？**
- **progress_reward 太弱**（mean=0.016）：主学习信号太弱，无法对抗环境内在的负奖励，也无法提供足够梯度。这是根本原因。
- **soft_landing_proxy 几乎死代码**（nonzero_rate=0.3%）：`next_obs[6] == 1.0` 使用精确浮点比较，连续环境中几乎永远不会触发 → 这也是验证失败的根因。
- **stability_penalty** ratio=-0.378，已在上轮降低，不是主要矛盾。

**3. 我之前改了什么？**
Iter 2 把 stability_penalty 系数降了 10 倍，得分反而从 -88 降到 -109。说明瓶颈不在 stability，而在信号强度。

**验证失败原因：** `next_obs[6] == 1.0` / `next_obs[7] == 1.0` 是精确浮点相等比较，连续 lunar lander 的 leg contact 是连续值，这个条件几乎永远为 False。验证框架检测到了这条死代码路径。

## 修改方案（层次 2：改数学形式）

三个改动，聚焦一个信号（progress）为主：

1. **progress_reward ×8**：从 scale=1 → scale=8（skeleton 推荐 5~20），预期 mean 从 0.016 → ~0.13
2. **soft_landing_proxy 从二值 → 连续乘积因子**：用 `max(0, 1-x/threshold)` 取代 `==` 比较，提供梯度
3. **stage weighting**：早期训练完全忽略 stability_penalty，后期逐步加入，避免早期压制探索

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

    # 3. 软着陆近似信号：连续乘积因子替代二值条件
    # 每个因子 bounded 在 [0,1]，提供梯度
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

**预期效果：**
- `progress_reward` mean 从 0.016 → ~0.13（8 倍增强），能对抗环境负奖励
- `stability_penalty` ratio 自然缩小到 ~-0.05（因为 progress 分母变大），且早期为 0
- `soft_landing_proxy` 从死代码变成连续梯度信号，nonzero_rate 大幅提升
- 不再使用 `==` 浮点比较，验证应通过
