## 分析

### 1. 发生了什么？
**着陆奖励从未触发（active_rate=0%）**，这是问题根源。

证据链：
- `landing_proxy` 的 `active_rate=0.0%`，`episode_sum_mean=0.0` — 在整个20个评估回合中，没有任何一步满足着陆条件。
- `terminated=4/20, truncated=16/20, episode_length=817.55` — 80%的回合被时间截断，说明智能体没有完成着陆任务。
- `proximity_reward` 占 magnitude_share 的 73.2%，每回合累积 52.68 — 智能体学会了靠近目标区域悬停，大量获取 proximity 奖励。
- `progress_reward` 仅累积 7.93（在权重 8.0 下），说明净位移仅约 1 单位 — 智能体在目标附近徘徊，不再前进。

因果链：当前代码的 `contact_factor` 在没有支撑腿接触时为 0，导致 `r_landing = 30.0 * ... * 0 = 0` 始终为零。阈值又极紧（dist<0.3, speed<0.2, angle<0.15），即使接触也需要近乎完美姿态。结果：着陆信号完全缺失 → 智能体从未收到"需要降落"的梯度 → 学会在目标上方悬停吃 proximity 奖励。

### 2. 最值得干预的组件？
**`landing_proxy`**。这是唯一的零激活组件，且与任务目标（降落）直接对应。数学形态上，其乘积结构中的 `contact_factor`（离散值 0/0.15/1.0）是硬门控——没有接触就归零，完全切断了着陆前的梯度。历史最佳代码使用 `0.1 + 0.9*contact` 保留了底值，但阈值仍偏紧、着陆信号偏弱，不足以对抗 proximity 的悬停诱惑。

### 3. 上一轮改了什么？
从 Agent Memory 可见：上一轮基于最佳代码做了三个改动——
- 将主信号从 `r_dist = 1/(1+d)` 换成 `r_progress = 8*(old-new)` + 弱 `r_proximity = 0.1/(1+d)`
- 将 landing 阈值大幅收紧（0.5→0.3, 0.4→0.2, 0.3→0.15）
- 将 contact 从连续梯度 `(0.1+0.9*contact)` 改为离散门控

效果：landing 从可能低激活变为零激活，得分暴跌。

### 修改方案
以最佳代码为基底，做三个有证据的修改：
1. **着陆 proxy 去门控化**：`contact` 从门槛变为放大器 `(1.0 + contact_sum)`，确保无接触时也能获得梯度；放宽阈值，使着陆前行为可被奖励。
2. **引入适度 progress 项**：保留 `1/(1+d)` 作为基础 proximity，同时加入 `old_dist - new_dist` 奖励接近运动，对抗悬停。
3. **提升 landing 权重**：使着陆信号在接近目标时成为主导，引导智能体完成降落而非无限悬停。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current and next positions
    cx, cy = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    old_dist = (cx ** 2 + cy ** 2) ** 0.5
    new_dist = (nx ** 2 + ny ** 2) ** 0.5

    # 1. Progress: reward each step toward target, penalize retreat
    r_progress = 2.0 * (old_dist - new_dist)

    # 2. Proximity: mild always-on gradient toward (0,0)
    r_proximity = 0.5 / (1.0 + new_dist)

    # 3. Stability penalty: discourage violent motion
    r_stability = -(
        0.01 * (abs(nvx) + abs(nvy)) +
        0.1 * abs(body_angle) +
        0.05 * abs(ang_vel)
    )

    # 4. Soft landing proxy: wide thresholds, contact amplifies but never gates
    proximity = max(0.0, 1.0 - new_dist / 0.8)
    stillness = max(0.0, 1.0 - (abs(nvx) + abs(nvy)) / 0.6)
    upright   = max(0.0, 1.0 - abs(body_angle) / 0.5)
    contact_sum = left_contact + right_contact  # 0.0, 1.0, or 2.0

    r_landing = 15.0 * proximity * stillness * upright * (1.0 + contact_sum)

    total_reward = r_progress + r_proximity + r_stability + r_landing

    components = {
        "progress_reward": r_progress,
        "proximity_reward": r_proximity,
        "stability_penalty": r_stability,
        "landing_proxy": r_landing
    }

    return float(total_reward), components
```