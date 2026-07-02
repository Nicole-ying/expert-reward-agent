## 诊断

### 1. 这个 agent 发生了什么？

Episode length = 1000（完整），没有 crash。但得分从 141.58 暴跌到 66.52。不是 crash，不是徘徊——是**着陆质量显著下降**。agent 仍然在飞完整个 episode，但着陆不够好。

### 2. 哪个组件是主要原因？

| 组件 | iter3(best) mean | iter4(now) mean | 变化 |
|------|-----------------|-----------------|------|
| soft_landing_proxy | 0.221 | 0.077 | **-65%** |
| progress_reward | 0.046 | 0.045 | 持平 |

soft_landing_proxy 从每步持续奖励（binary 0.5，nonzero_rate ~44%）变成了事件触发（just_landed, nonzero_rate 7.7%）。总着陆激励缩水了约 65%。progress_reward 和 stability_penalty 在 iter 3→4 之间没变，排除它们是根因。

### 3. 我之前改了什么？

上一轮把 soft_landing_proxy 从"每步状态奖励"改成"一次性触地事件奖励"，理由是担心 agent 静坐收割。但 iter 2-3 的 141-142 分表明：**静坐收割不是问题，agent 确实在好好着陆**。事件型改动反而摧毁了密集梯度，得分暴跌 75 分。

### 修复方案

**层次 2：改数学形式。** 回到每步奖励，但用连续 bounded 形式替代 binary if-else。`max(0, 1-x/D)` 为每个着陆条件提供平滑梯度，奖励随质量连续缩放——比 binary 更难 exploit，同时恢复密集信号。

只改 soft_landing_proxy 一个组件。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 目标位置 (0,0)，obs 中 x,y 是相对于目标平台的坐标
    target_pos = (0.0, 0.0)

    # 1. 主学习信号：进度奖励（保持 scale=10，iter 3-4 均验证有效）
    dist = ((obs[0] - target_pos[0]) ** 2 + (obs[1] - target_pos[1]) ** 2) ** 0.5
    next_dist = ((next_obs[0] - target_pos[0]) ** 2 + (next_obs[1] - target_pos[1]) ** 2) ** 0.5
    progress_reward = (dist - next_dist) * 10.0

    # 2. 稳定性惩罚（保持不变，ratio=-0.014 已是无害背景信号）
    vel_x = abs(next_obs[2])
    vel_y = abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    stability_penalty = -0.001 * (vel_x + vel_y) - 0.001 * angle - 0.001 * ang_vel

    # 3. 软着陆 proxy：从事件型回到每步连续型
    #    诊断：事件型 (just_landed) 使着陆激励缩水 65%，得分暴跌。
    #    修复：用 bounded 连续因子 max(0, 1-x/D) 替代 binary if-else，
    #    每个着陆条件独立贡献梯度，奖励随质量连续缩放。
    #    contact 也用连续值（乘积）替代 binary gate，为双脚接近地面提供梯度。

    near_factor = max(0.0, 1.0 - next_dist / 0.3)
    speed = abs(next_obs[2]) + abs(next_obs[3])
    speed_factor = max(0.0, 1.0 - speed / 0.5)
    angle_factor = max(0.0, 1.0 - abs(next_obs[4]) / 0.2)

    # 连续接触因子：双脚乘积，提供"双脚同时着地"的梯度
    left_contact = min(1.0, max(0.0, next_obs[6]))
    right_contact = min(1.0, max(0.0, next_obs[7]))
    contact_factor = left_contact * right_contact

    # 四个因子相乘：任何维度偏离最佳状态都会降低奖励
    soft_landing_proxy = near_factor * speed_factor * angle_factor * contact_factor * 0.5

    total_reward = progress_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }
    return float(total_reward), components
```