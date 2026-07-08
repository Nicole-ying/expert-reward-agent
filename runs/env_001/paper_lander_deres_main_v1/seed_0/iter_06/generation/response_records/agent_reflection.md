# Response Record

## 诊断分析

### 0. 环境理解
2D 飞行器着陆任务：从上方出发，到达原点 (0,0) 并平稳着陆（双腿着地、低速、直立）。观测 8 维，离散动作 4 个（无引擎/左姿态/主引擎/右姿态）。

### 1. Agent 发生了什么？
- **episode_length=1000（满）**，说明 agent 没有 crash，但也没有高效完成任务——它在目标附近**徘徊/悬浮**。
- score=139.74，比 iter 3 最佳（146.36）低约 6.6 分。
- `progress_reward` mean 仅 0.002，几乎为零——agent 移动极少。
- `landing_proxy` nonzero_rate=62%，ratio=63.81——这个信号极大且持续发放，鼓励 agent 维持双腿着地状态而不触发 settled 终止。

### 2. 哪个组件是主要原因？
**landing_proxy** 是问题根源。它每步发放约 0.25 奖励（只要双腿着地），让 agent 学会"着陆后继续活跃以持续收割奖励"而非"尽快稳定以触发任务完成"。这解释了满 episode 长度和略低的得分。

### 3. 之前改了什么？
Iter 5 在 iter 3 骨架上添加了 `landing_proxy`，得分从 146.36 → 139.74（下降）。按照 revert 规则，应回到历史最佳 iter 3 骨架，并做一个新的修改。

### 验证失败
v6 代码验证失败的具体原因未给出，但去掉 `landing_proxy` 回到 iter 3 骨架后应该能解决。

### 修改方案
**回到 iter 3 骨架 + 降低 w_approach（0.15 → 0.10）**：进一步削弱 hovering/徘徊的激励，让 progress 信号相对更强，促使 agent 更主动地向目标移动。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current and next states
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_angle = next_obs[4]

    # 1. Core learning signal: progress towards target (0,0)
    dist_current = (x ** 2 + y ** 2) ** 0.5
    dist_next = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = dist_current - dist_next  # positive when approaching

    # 2. Continuous approach quality shaping
    #    Three bounded factors [0,1], product ensures all must be satisfied
    prox_factor = 1.0 / (1.0 + 5.0 * dist_next)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    speed_factor = 1.0 / (1.0 + speed)
    angle_factor = 1.0 / (1.0 + 5.0 * abs(next_angle))
    approach_quality = prox_factor * speed_factor * angle_factor  # [0, 1], dense

    # 3. Orientation penalty (gentle, keeps the craft upright)
    attitude_penalty = -0.01 * abs(next_angle)

    # w_approach reduced from 0.15 to 0.10: further curb hovering,
    # making progress the relatively stronger signal to encourage movement.
    w_approach = 0.10
    approach_quality_reward = w_approach * approach_quality
    total_reward = progress + approach_quality_reward + attitude_penalty

    components = {
        "progress_reward": progress,
        "approach_quality_reward": approach_quality_reward,
        "attitude_penalty": attitude_penalty,
    }

    return float(total_reward), components
```
