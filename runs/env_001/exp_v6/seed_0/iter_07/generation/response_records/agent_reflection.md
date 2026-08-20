# Response Record

## 诊断

### 1. Agent 发生了什么？

Episodes 全部跑满 1000 步，没有 crash。得分在 141-146 之间徘徊 6 轮无法突破。`soft_landing_proxy` 的 nonzero_rate 高达 **42.9%**（一个"稀缺事件型"信号在近半时间触发），ratio_to_progress = **51.76x**。Agent 在利用 proxy 条件漏洞：它学会了在目标附近悬停（满足 near + low_speed + stable_angle + both_contact），每步收割 0.5 奖励，而不是真正完成着陆。`progress_reward`（mean=0.004）在总信号中占比不到 2%，形同虚设。

### 2. 哪个组件是主因？

**`soft_landing_proxy`**。二值条件 + 0.5 高额奖励 + 条件过松 → 触发率 43%，均值 0.214，是 progress 的 51 倍。Agent 被 proxy 支配，优化的是"满足条件悬停"而非"着陆"。

### 3. 我上一轮改了什么？

Iter 6 的代码与 Iter 2 完全相同（得分也完全一样 141.58）。之前 Iter 3-5 尝试过加大 progress_reward（0.045），但 proxy ratio 依然 ~5:1，天花板仍在 146。**单纯调系数已经试过 2+ 轮，无法突破**，需要改数学形态。

### 方案：层次 2 — 把二值 proxy 改为连续 bounded 乘积

- 每个条件从 `if` 硬阈值 → `1/(1+kx)` 连续衰减，全程有梯度
- 乘积形式保证"所有条件同时好才高分"，天然 anti-exploit
- 系数从 0.5 降到 0.1，max 缩至原来的 1/5，让 progress 重新成为主导信号

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

    # 3. 连续软着陆 proxy — 从二值改为 bounded 乘积，全程梯度 + 自动 anti-exploit
    # near_factor: 1/(1+10*dist), dist=0→1.0, dist=0.1→0.5, dist=0.5→0.17
    near_factor = 1.0 / (1.0 + 10.0 * next_dist)
    # speed_factor: 1/(1+5*speed), speed=0→1.0, speed=0.2→0.5, speed=0.5→0.29
    speed_factor = 1.0 / (1.0 + 5.0 * (vel_x + vel_y))
    # angle_factor: 1/(1+10*angle), angle=0→1.0, angle=0.1→0.5, angle=0.3→0.25
    angle_factor = 1.0 / (1.0 + 10.0 * angle)
    # contact_factor: 连续 0~1，两条腿平均接触
    contact_factor = (next_obs[6] + next_obs[7]) / 2.0

    soft_landing_proxy = 0.1 * near_factor * speed_factor * angle_factor * contact_factor

    total_reward = progress_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }
    return float(total_reward), components
```

**预期效果**：
- `soft_landing_proxy` max 从 0.5 降至 0.1，且只在所有条件同时优秀时才接近 max
- 典型悬停场景（dist≈0.05, speed≈0.15, angle≈0.05, contact≈0.7）下 proxy ≈ 0.67×0.57×0.67×0.7×0.1 ≈ **0.018**，约 progress 的 4-5 倍（远优于之前的 51 倍）
- 连续梯度让 agent 可以在"接近着陆条件"时就获得递增信号，而非二值的全有/全无
- 乘积形式天然要求所有条件同时满足，单个条件松懈不足以刷分
