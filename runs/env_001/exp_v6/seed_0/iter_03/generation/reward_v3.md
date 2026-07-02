## 诊断

### 1. Agent 发生了什么？
- **episode_length=1000（满）**，不再 crash，score=141，中等水平。
- **soft_landing_proxy 主导了总奖励**：ratio_to_progress = 51.76，占总奖励的 98%。agent 每步 ~43% 触发 0.5 的 bonus，这意味着它在目标平台附近大量徘徊/停留，靠反复触发 soft_landing_proxy 赚取大部分奖励。
- **progress_reward 几乎消失**：mean 仅 0.004，在目标附近距离变化微小时自然很小，但这也意味着它无法提供有意义的学习梯度。

### 2. 哪个组件是主要原因？
- **progress_reward 太弱**：mean=0.004，skeleton 建议 scale 5~20，当前无 scale。这是根因——主学习信号过弱，agent 自然依赖 bonus。
- soft_landing_proxy 的 ratio 高是**症状而非根因**：bonus 类天然 ratio 高，但如果 progress 够强，bonus 不会淹没学习。nonzero_rate 43% 偏高但可接受。

### 3. 我之前改了什么？
- 上一轮把 stability_penalty 从 0.01 降到 0.001，效果极好：episode 从 74→1000，score 从 -112→141。
- 本轮应聚焦**增强主信号**，不重复调整 stability。

### 修改方案
**层次 1：改系数**——给 progress_reward 加 scale=10。预计 progress mean ~0.04，soft/progress ratio 从 52→~5，progress 重新成为主导梯度来源。不改 soft_landing_proxy（下轮再观察其 nonzero_rate 变化）。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 目标位置 (0,0)，因为 obs 中 x,y 是相对于目标平台的坐标
    target_pos = (0.0, 0.0)

    # 1. 主学习信号：基于距离变化的进度奖励
    #    上轮 mean=0.004 过弱，导致 soft_landing_proxy (ratio=51.76) 主导总奖励。
    #    本轮加 scale=10（skeleton 建议 5~20），预计 progress mean ~0.04，
    #    使 soft/progress ratio 从 52 降至 ~5，恢复 progress 的主导地位。
    dist = ((obs[0] - target_pos[0]) ** 2 + (obs[1] - target_pos[1]) ** 2) ** 0.5
    next_dist = ((next_obs[0] - target_pos[0]) ** 2 + (next_obs[1] - target_pos[1]) ** 2) ** 0.5
    progress_reward = (dist - next_dist) * 10.0  # 加 scale，每一步靠近目标获得正奖励

    # 2. 稳定性惩罚（保持不变）
    vel_x = abs(next_obs[2])
    vel_y = abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    stability_penalty = -0.001 * (vel_x + vel_y) - 0.001 * angle - 0.001 * ang_vel

    # 3. 软着陆 proxy（本轮未改，留待下轮观察）
    near_target = (next_dist < 0.1)
    low_speed = (vel_x + vel_y < 0.2)
    stable_angle = (angle < 0.1)
    both_contact = (next_obs[6] > 0.5 and next_obs[7] > 0.5)
    soft_landing_proxy = 0.5 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    total_reward = progress_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }
    return float(total_reward), components
```