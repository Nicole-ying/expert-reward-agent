## 诊断

**1. Agent 发生了什么？**
Episode 长度仅 69 步，100% 提前终止，得分 -111.7。Agent 在 crash，没有学会着陆。

**2. 哪个组件是主要原因？**
- **landing_proxy**: nonzero_rate = 0.46%，几乎从未触发。二值条件 `dist<0.3 AND speed<0.3 AND angle<0.2 AND legs touching` 对 agent 来说是不可逾越的门槛——它在 crash 之前永远满足不了这些条件。agent 没有终端引导信号，不知道"正确着陆"长什么样。
- **stability_penalty**: ratio_to_progress = -0.703，惩罚是主信号的 70%，过于强势。但根本问题是 agent 没有目标——如果它知道要去哪、怎么着陆，适度的稳定性惩罚反而是帮助。

**3. 之前改了什么？**
这是第 1 轮，没有历史。当前骨架的三个组件从零开始。

**策略**：聚焦 **landing_proxy** 的数学形式。从硬性二值条件改为连续乘积因子（bounded_continuous_proxy 技法），让 agent 在接近着陆垫的整个过程中都收到梯度信号。只改这一个组件，其他系数不动。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Calculate distance to target (0,0) for obs and next_obs
    dist_obs = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # Primary learning signal: progress towards the landing pad
    progress_reward = 1.0 * (dist_obs - dist_next)

    # Stability penalty: discourage high speeds, large angle, and angular velocity in the new state
    speed_next = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    stability_penalty = -0.01 * speed_next - 0.01 * abs(next_obs[4]) - 0.005 * abs(next_obs[5])

    # Continuous landing proxy: product of bounded factors provides gradient throughout approach.
    # Each factor max(0, 1 - value/threshold) decays linearly from 1→0 as the dimension worsens.
    # This replaces the binary if-condition that had 0.46% trigger rate with a signal that
    # activates as soon as the agent enters the approach zone.
    near_factor = max(0.0, 1.0 - dist_next / 1.0)       # active within 1.0 units of pad
    slow_factor = max(0.0, 1.0 - speed_next / 1.0)      # rewards speed < 1.0
    upright_factor = max(0.0, 1.0 - abs(next_obs[4]) / 0.5)  # rewards |angle| < 0.5
    leg_factor = (next_obs[6] + next_obs[7]) / 2.0       # both legs down = 1.0

    landing_proxy = near_factor * slow_factor * upright_factor * leg_factor

    total_reward = progress_reward + stability_penalty + landing_proxy

    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'landing_proxy': landing_proxy,
        'total_reward': total_reward
    }
    return float(total_reward), components
```

### 改动说明

| 维度 | 旧（二值 if） | 新（连续乘积） |
|---|---|---|
| **near_factor** | `dist_next < 0.3` (0/1) | `max(0, 1 - dist_next/1.0)` (0→1 渐变) |
| **slow_factor** | `speed_next < 0.3` (0/1) | `max(0, 1 - speed_next/1.0)` (0→1 渐变) |
| **upright_factor** | `abs(angle) < 0.2` (0/1) | `max(0, 1 - |angle|/0.5)` (0→1 渐变) |
| **leg_factor** | 二值 AND 条件的一部分 | `(leg_left + leg_right)/2` (0→1 渐变) |

**关键效果**：旧代码中 landing_proxy 的 nonzero_rate 是 0.46%（只有完美着陆瞬间才触发）。新代码中，只要 agent 进入距目标 1.0 单位的范围内，near_factor > 0，landing_proxy > 0，nonzero_rate 将接近 100%。agent 在整个接近过程中都能感知到"靠近、减速、摆正"的梯度引导。