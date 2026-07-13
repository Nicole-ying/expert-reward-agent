# Response Record

## 分析

### 1. Agent 发生了什么？

**核心问题：agent 从未接近着陆条件，始终徘徊在中距离。**

- **`landing_proxy` active_rate = 0.0%**：所有20条评估轨迹中，着陆代理奖励从未触发过一次。三个阈值条件（距离<0.8、速度和<0.6、角度<0.5）从未被同时满足。agent 学到的策略是保持在离目标约 1.1~1.2 距离处徘徊，而非真正接近并着陆。
- **`progress_reward` 几乎为零**：1000步累计仅 0.83（每步 ~0.0008），说明 agent 没有形成向目标移动的趋势。old_dist 和 new_dist 几乎相等，agent 在绕圈或悬停。
- **`proximity_reward` 占绝对主导**（magnitude_share 97.1%，每步 ~0.234）：`0.5/(1+new_dist)` 在中等距离创造了舒适的奖励高原，agent 只需保持在距离≈1.14 附近就能稳定获取奖励，没有动力进一步接近目标。
- **全部 episode 被截断**（20/20 truncated, length=1000）：agent 从未触发任何终止条件（成功着陆或坠毁），在时间上限耗尽前一直在徘徊。

**根因**：常驻 proximity_reward 制造了"舒适区"，progress_reward 强度不足以打破它，landing_proxy 阈值过紧导致零梯度。

### 2. 最值得干预的组件

**`proximity_reward`（常驻距离奖励）**——它是问题的根源。它提供了破坏性的反向激励：agent 学会停在中等距离吃稳定收益，而非继续接近目标。需要移除或改用 potential-based 差分形式，使奖励只来自"向目标移动"而非"待在某个位置"。

配合修改 **`landing_proxy` 阈值**——当前阈值（距离<0.8、速度和<0.6、角度<0.5）太紧，agent 从未触及。需大幅放宽，确保早期探索就能获得非零梯度。

### 3. 上一轮改动回顾

上轮在 best 基础上做了三处改动：
- 将常驻 `r_dist`（系数 1.0）削弱为 `r_proximity`（系数 0.5），并增加 `r_progress = 2.0*(old_dist - new_dist)`
- 放宽 landing 阈值：距离 0.5→0.8，速度 0.4→0.6，角度 0.3→0.5
- 提升 landing 系数：8.0→15.0

**实际效果**：proximity 仍占 97% 统治地位，progress 微弱到可忽略（0.3% signed_share），landing 仍为 0%。改动方向正确（加 progress、放宽阈值），但力度远远不够——proximity 依然制造舒适区，progress 系数 2.0 太小。

### 修改方案

1. **彻底移除常驻 proximity_reward**，改用 potential-based shaping：`r_progress = scale * (potential(next) - potential(current))`，其中 `potential = 1/(1+distance)`。这保证奖励只来自实际靠近目标的行为，不奖励停滞。
2. **大幅提升 progress 系数**至 15.0，使其成为主驱动信号。
3. **大幅放宽 landing_proxy 阈值**：距离<1.5、速度和<1.2、角度<1.0，确保在较宽区域内产生非零梯度，引导 agent 向精确着陆条件收缩。
4. 保留 stability_penalty 不变。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    cx, cy = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    old_dist = (cx ** 2 + cy ** 2) ** 0.5
    new_dist = (nx ** 2 + ny ** 2) ** 0.5

    # 1. Potential-based progress: only rewards movement toward target
    #    potential = 1/(1+d) is bounded in [0,1], gradient strongest near target
    old_potential = 1.0 / (1.0 + old_dist)
    new_potential = 1.0 / (1.0 + new_dist)
    r_progress = 15.0 * (new_potential - old_potential)

    # 2. Stability penalty: discourage violent motion
    r_stability = -(
        0.01 * (abs(nvx) + abs(nvy)) +
        0.1 * abs(body_angle) +
        0.05 * abs(ang_vel)
    )

    # 3. Soft landing proxy: wide thresholds to guarantee early gradient
    #    Each factor is max(0, 1 - value/threshold), activates gradually
    proximity = max(0.0, 1.0 - new_dist / 1.5)
    stillness = max(0.0, 1.0 - (abs(nvx) + abs(nvy)) / 1.2)
    upright   = max(0.0, 1.0 - abs(body_angle) / 1.0)
    contact   = (left_contact + right_contact) / 2.0
    r_landing = 10.0 * proximity * stillness * upright * (0.1 + 0.9 * contact)

    total_reward = r_progress + r_stability + r_landing

    components = {
        "progress_reward": r_progress,
        "stability_penalty": r_stability,
        "landing_proxy": r_landing
    }

    return float(total_reward), components
```
