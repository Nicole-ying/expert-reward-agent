# Response Record

## 分析

### 1. 这个 agent 发生了什么？

**核心问题：distance_reward 是"存活奖励"，agent 学会了在目标垫附近悬停而不是降落。**

- 全部 20 个 episode 被截断（truncated=20/20），无一次自然终止（terminated=0/20）。agent 从未成功着陆。
- `distance_reward = 1/(1+distance)` 持续触发（active_rate=100%），每步给 ~0.73 奖励，1000 步累计 ~728。数学形态是**持续状态值**——只要待在目标附近就有正奖励，不要求改变状态或完成任务。
- `landing_bonus` 的 active_rate=0%，硬性二值条件（distance<0.2 且低速且小角度且双腿接触）从未满足，完全无法提供学习引导。
- `stability_penalty` 均值仅 -6.14，相对于 728 的距离奖励微不足道，无法对抗悬停策略。
- 得分范围 [-44.9, 17.1]，外部评分系统对"未着陆"给予负分，但我的奖励函数给悬停行为大量正反馈——典型 proxy misalignment。

### 2. 哪个组件最值得干预？

**distance_reward 的数学形态**（持久状态值 `1/(1+d)`）是根源。检索到的 `state_to_improvement` 变换明确指出：应把持久状态值替换为**改进量**（potential-based shaping），使奖励来自"变得更好"而非"占据某个状态"。同时 `bounded_continuous_proxy` 指出硬性二值 landing_bonus 需要改成连续乘积形式。

### 3. 我之前改了什么？

上一轮（也是当前唯一一轮）使用持久距离奖励 + 极小稳定性惩罚 + 不可达的二值着陆奖励。没有时间惩罚，导致悬停成为最优策略。

### 修改方案

1. **距离奖励 → potential-based progress**：`10 * (distance_current - distance_next)`，奖励接近目标、惩罚远离，总奖励有界（初始距离×10），悬停无收益。
2. **二值 landing_bonus → 连续乘积 proxy**：`proximity × stillness × upright × contact_factor`，每项用 `max(0, 1-x/threshold)` bounded 形式，contact_factor 设 0.15 地板确保无接触时也有微弱信号。
3. **新增时间惩罚**：`-0.02/step`，激励快速完成，防止拖延。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current state
    x_pos = obs[0]
    y_pos = obs[1]

    # Next state
    x_pos_next = next_obs[0]
    y_pos_next = next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    body_angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Potential-based progress: reward getting closer, penalize moving away
    #    No reward for staying still. Total over episode is bounded by initial distance.
    dist_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    dist_next = (x_pos_next ** 2 + y_pos_next ** 2) ** 0.5
    r_progress = 10.0 * (dist_current - dist_next)

    # 2. Continuous landing proxy: replaces hard binary bonus with smooth gradient
    #    Each factor is bounded in [0, 1]; product gives dense learning signal.
    proximity = max(0.0, 1.0 - dist_next / 0.3)
    stillness = max(0.0, 1.0 - (abs(x_vel_next) + abs(y_vel_next)) / 0.5)
    upright = max(0.0, 1.0 - abs(body_angle_next) / 0.3)
    # Contact factor: 0.15 floor ensures weak signal before any leg contact;
    # one leg = 0.575, both legs = 1.0 — strong gradient toward touchdown.
    contact_factor = 0.15 + 0.85 * (left_contact + right_contact) / 2.0
    r_landing = 30.0 * proximity * stillness * upright * contact_factor

    # 3. Time penalty: discourage hovering, incentivize fast completion
    r_time = -0.02

    total_reward = r_progress + r_landing + r_time

    components = {
        "progress_reward": r_progress,
        "landing_proxy": r_landing,
        "time_penalty": r_time
    }

    return float(total_reward), components
```
