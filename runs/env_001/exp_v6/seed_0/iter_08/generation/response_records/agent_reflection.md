# Response Record

## 诊断

### 1. 发生了什么？

Score 从 146.13（iter 5，binary 版本）降到 124.97（iter 7，连续乘积版本）。Episode 长度 919，没有 crash，但得分退步了。

核心问题：**soft_landing_proxy 从二值改为连续乘积后，ratio_to_progress 飙到 12.47，nonzero_rate 63.5%。** agent 学会了持续触发 proxy 的部分分数，而不是真正完成着陆——这就是"proxy hacking"。

### 2. 哪个组件是主因？

`soft_landing_proxy`。连续乘积形态下，`near_factor = 1/(1+10*dist)` 衰减太慢（dist=0.3 时仍为 0.25），导致 agent 在较远距离就能获得可观的 proxy 奖励。四个因子乘积后均值仍有 0.046，远超 progress_reward 的 0.0037。

### 3. 我上次改了什么？

Iter 6→7：把二值 soft_landing_proxy 改成连续乘积。效果变差。所以这轮不应该继续在"连续乘积"这条路上硬调——需要改变策略：**保留连续梯度，但用更尖锐的距离门控让 proxy 只在真正靠近时才激活。**

### 修改方案

- 把 `near_factor = 1/(1+10*dist)` 换成 `dist_gate = 1/(1+100*dist²)`。平方+大系数让 gate 在 dist>0.2 时迅速趋零，只保留真正近距离的梯度。
- 降低 multiplier 从 0.1 到 0.05，进一步控制 ratio。
- progress_reward 和 stability_penalty 不动——它们不是问题根源。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 目标位置 (0,0)，obs 中 x,y 是相对于目标平台的坐标
    target_pos = (0.0, 0.0)

    # 1. 主学习信号：基于距离变化的进度奖励（保持不变）
    dist = ((obs[0] - target_pos[0]) ** 2 + (obs[1] - target_pos[1]) ** 2) ** 0.5
    next_dist = ((next_obs[0] - target_pos[0]) ** 2 + (next_obs[1] - target_pos[1]) ** 2) ** 0.5
    progress_reward = dist - next_dist

    # 2. 稳定性惩罚（保持不变）
    vel_x = abs(next_obs[2])
    vel_y = abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    stability_penalty = -0.001 * (vel_x + vel_y) - 0.001 * angle - 0.001 * ang_vel

    # 3. 连续软着陆 proxy — 尖锐距离门控 + 连续梯度
    # dist_gate: 1/(1+100*dist^2), dist=0.1→0.5, dist=0.2→0.2, dist=0.3→0.1
    # 只在真正接近目标时激活，避免远距离 proxy hacking
    dist_gate = 1.0 / (1.0 + 100.0 * next_dist ** 2)
    # speed_factor: 1/(1+5*speed), speed=0→1.0, speed=0.2→0.5
    speed_factor = 1.0 / (1.0 + 5.0 * (vel_x + vel_y))
    # angle_factor: 1/(1+10*angle), angle=0→1.0, angle=0.1→0.5
    angle_factor = 1.0 / (1.0 + 10.0 * angle)
    # contact_factor: 连续 0~1，两条腿平均接触
    contact_factor = (next_obs[6] + next_obs[7]) / 2.0

    soft_landing_proxy = 0.05 * dist_gate * speed_factor * angle_factor * contact_factor

    total_reward = progress_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }
    return float(total_reward), components
```
