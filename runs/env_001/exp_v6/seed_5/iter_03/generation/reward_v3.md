## 诊断

### 1. Agent 发生了什么？
Episode 长度 70 步，100% 早停（crash）。得分从 -111.71 改善到 -89.18，说明上轮的连续 landing_proxy 替换二值条件有正向效果，但 agent 仍然在撞击。

### 2. 哪个组件是主要原因？

- **stability_penalty**：ratio_to_progress = -0.71（绝对值 > 0.5），属于 `stability_penalty_dominance` 失败模式。惩罚信号是主学习信号的 71%，agent 在"向目标移动"和"别动"之间被撕裂。nonzero_rate=100%，意味着每一步都在被罚。
- **landing_proxy**：nonzero_rate=1.1%，仍然极稀疏。连续乘积形态虽好于二值条件，但四个 [0,1] 因子相乘，大部分时间至少一个因子为 0。

根因判断：**stability_penalty 全局生效且过强是第一优先问题**。它在远离目标时也在惩罚速度和角度，直接与 progress_reward 冲突。知识库建议：降低 10 倍或距离门控。

### 3. 我之前改了什么？
上轮把二值 landing 条件改成连续乘积，得分从 -111.71 → -89.18（有效）。本轮聚焦 stability_penalty，不再动 landing_proxy。

---

**修改方案**：给 stability_penalty 加距离门控。远离目标时惩罚趋零，让 agent 自由移动；靠近目标时惩罚逐渐激活，引导减速和稳定。用 `1/(1+5*dist)` 做平滑门控——这是 bounded 形式，值域天然在 [0,1]，不需要和 progress 的量级对齐。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Calculate distance to target (0,0) for obs and next_obs
    dist_obs = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # Primary learning signal: progress towards the landing pad
    progress_reward = 1.0 * (dist_obs - dist_next)

    # Stability penalty with distance-gating.
    # Gate = 1/(1+5*dist): ~1.0 at pad, ~0.17 at dist=1.0, ~0.05 at dist=4.0.
    # This lets the agent move freely when far away, and only tightens
    # speed/angle/angvel constraints as it approaches the landing zone.
    speed_next = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    dist_gate = 1.0 / (1.0 + 5.0 * dist_next)
    stability_penalty = dist_gate * (
        -0.01 * speed_next
        - 0.01 * abs(next_obs[4])
        - 0.005 * abs(next_obs[5])
    )

    # Continuous landing proxy: product of bounded factors provides gradient
    # throughout approach. Each factor max(0, 1 - value/threshold) decays
    # linearly from 1→0 as the dimension worsens.
    near_factor = max(0.0, 1.0 - dist_next / 1.0)
    slow_factor = max(0.0, 1.0 - speed_next / 1.0)
    upright_factor = max(0.0, 1.0 - abs(next_obs[4]) / 0.5)
    leg_factor = (next_obs[6] + next_obs[7]) / 2.0

    landing_proxy = near_factor * slow_factor * upright_factor * leg_factor

    total_reward = progress_reward + stability_penalty + landing_proxy

    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'dist_gate': dist_gate,
        'landing_proxy': landing_proxy,
        'total_reward': total_reward
    }
    return float(total_reward), components
```