## 分析

### 1. 这个 agent 发生了什么？

**核心问题：持久性奖励 farming，导致行为缓慢、无法完成任务。**

证据：
- **episode 长达 951 步**，14/20 被截断（超时），仅 6/20 成功终止。agent 在"存活但不着陆"的状态徘徊。
- **组件总和 vs 实际 score 严重偏离**：`distance_reward`(649) + `landing_proxy`(324) + `stability_penalty`(-6) ≈ **+967**，但实际 score = **-64.5**。差额约 -1031，对应环境隐式燃料惩罚（每步约 -1.08）。agent 没有动力加速完成。
- **当前版本拓宽了 landing_proxy 阈值**（proximity 从 0.5→1.0，stillness 从 0.4→0.8），使 proxy 更容易触发（active_rate 90%），但稀释了精确着陆的激励，进一步助长"徘徊收集 proxy 奖励"的行为。

**根因**：`distance_reward = 1/(1+d)` 是持久状态值——agent 只需待在距离目标不太远的位置就能持续获得正向奖励，完全没有时间压力。`state_to_improvement` 诊断直接命中此模式。

### 2. 最值得干预的组件

**`distance_reward`（占比 66.3%，100% 活跃）**。这是最大的持久性奖励源，也是 proxy farming 的主要驱动力。应将其从"占有好状态"改为"向好状态改进"，即从持久值变为进度增量。

`landing_proxy` 的阈值应恢复到 best 代码的紧版本（proximity < 0.5, stillness < 0.4），使其只在真正接近着陆时才激活。

### 3. 上一轮修改回顾

上一轮将 best 代码的 proximity 阈值从 0.5 放宽到 1.0、stillness 从 0.4 放宽到 0.8。意图是"提供更早梯度"，但实际效果是让 landing_proxy 更容易在非着陆状态下触发，配合持久 distance_reward，agent 学会了徘徊收集奖励而非完成任务。这是典型的 `proxy_to_completion_alignment` 失败。

### 修改方案

**核心变换：`state_to_improvement`**——将 distance_reward 从持久状态值改为进度增量 `max(0, old_dist - new_dist)`，仅当 agent 真正靠近目标时获得奖励。同时恢复 landing_proxy 紧阈值，降低 stability_penalty 系数以匹配新的奖励量级。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current state from obs
    cx_pos = obs[0]
    cy_pos = obs[1]
    
    # Next state from next_obs
    nx_pos = next_obs[0]
    ny_pos = next_obs[1]
    nx_vel = next_obs[2]
    ny_vel = next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    # 1. Progress-based distance reward (state_to_improvement transform)
    #    Only rewards actual progress toward the target — eliminates proxy farming.
    old_dist = (cx_pos ** 2 + cy_pos ** 2) ** 0.5
    new_dist = (nx_pos ** 2 + ny_pos ** 2) ** 0.5
    r_progress = 10.0 * max(0.0, old_dist - new_dist)
    
    # 2. Small persistent proximity for terminal gradient maintenance
    #    Keeps a weak pull toward target when progress stalls near zero.
    r_proximity = 0.3 / (1.0 + new_dist)
    
    # 3. Stability penalty — reduced coefficients to match new reward scale
    r_stability = -(
        0.005 * (abs(nx_vel) + abs(ny_vel)) +
        0.05 * abs(body_angle) +
        0.025 * abs(ang_vel)
    )
    
    # 4. Landing proxy with original tight thresholds (restored from best)
    #    Only activates when genuinely close to a soft landing.
    proximity = max(0.0, 1.0 - new_dist / 0.5)
    stillness = max(0.0, 1.0 - (abs(nx_vel) + abs(ny_vel)) / 0.4)
    upright   = max(0.0, 1.0 - abs(body_angle) / 0.3)
    contact   = (left_contact + right_contact) / 2.0
    r_landing = 15.0 * proximity * stillness * upright * (0.1 + 0.9 * contact)
    
    total_reward = r_progress + r_proximity + r_stability + r_landing
    
    components = {
        "progress_reward": r_progress,
        "proximity_reward": r_proximity,
        "stability_penalty": r_stability,
        "landing_proxy": r_landing
    }
    
    return float(total_reward), components
```