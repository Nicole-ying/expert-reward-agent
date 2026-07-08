## 诊断

### 1. Agent 状态：在徘徊/悬停，利用 landing proxy 但不终结 episode

- Episode 长度 922 step，得分 169.76，无 crash。
- `soft_landing_proxy` 的 nonzero_rate = 41.5%，mean = 0.294。这意味着约 383 步中 landing 条件被触发，且触发时平均值为 0.71（因子都接近 1）。
- Agent 在着陆区附近保持良好姿态，持续获得 landing proxy 奖励，但不实际着陆终结 episode。**典型 exploit 模式：悬停蹭分。**

### 2. 根因：soft_landing_proxy 严重主导

| 组件 | ratio_to_progress | 诊断 |
|---|---|---|
| `progress_reward` | 1.0（基准） | 正常，每步 0.043 |
| `soft_landing_proxy` | **6.9** | 主导，是 progress 的近 7 倍 |
| `stability_penalty` | ~0 | 完全死亡，nonzero_rate 仅 0.35% |

`original_env_reward` ratio = -7.4（与 progress 反向），说明 agent 在大量消耗燃料维持悬停——这正是 exploit landing proxy 的代价。

### 3. 上轮改了什么

Iter 3 引入 product-form landing proxy → 得分从 -108 跳到 169.76。但 proxy 权重未调过，它天然过强（4 因子乘积在满足时可达 1.0，远超 progress 的 0.043/step）。

**本轮策略：层次 1 — 给 soft_landing_proxy 加衰减系数 0.25。** 将其 ratio 从 6.9 压到 ~1.7，让 progress 和 landing 信号更平衡，减少悬停 exploit 的激励。不碰 stability_penalty（它已经死了，且历史上 active penalty 反而有害）。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 位置
    cx, cy = float(obs[0]), float(obs[1])
    nx, ny = float(next_obs[0]), float(next_obs[1])

    d_curr = (cx ** 2 + cy ** 2) ** 0.5
    d_next = (nx ** 2 + ny ** 2) ** 0.5

    # 1. 主学习信号：逐步靠近目标（scale=8，符合 skeleton 推荐 5~20）
    progress_reward = (d_curr - d_next) * 8.0

    # 2. 轻量稳定约束（当前 dead，保留结构不加改动）
    vx, vy = float(next_obs[2]), float(next_obs[3])
    speed = (vx ** 2 + vy ** 2) ** 0.5
    angle = abs(float(next_obs[4]))
    angular_v = abs(float(next_obs[5]))

    raw_stability_penalty = (
        -0.005 * speed
        - 0.01 * angle
        - 0.005 * angular_v
    )

    t = training_progress
    late_weight = max(0.0, min(1.0, 2.0 * t - 0.5))
    stability_penalty = raw_stability_penalty * late_weight

    # 3. 软着陆近似信号：连续乘积因子 + 衰减系数 0.25
    #    ratio_to_progress 从 6.9 → 约 1.7，减少悬停 exploit 激励
    proximity_factor = max(0.0, 1.0 - d_next / 0.3)
    speed_factor = max(0.0, 1.0 - speed / 0.5)
    angle_factor = max(0.0, 1.0 - angle / 0.3)
    contact_factor = (
        max(0.0, float(next_obs[6])) * max(0.0, float(next_obs[7]))
    )

    soft_landing_proxy = 0.25 * proximity_factor * speed_factor * angle_factor * contact_factor

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