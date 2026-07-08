# Response Record

## 诊断

### 0. 环境理解
2D 飞行器着陆任务。飞行器从上方出发，需到达 (0,0) 平台并用支撑腿接触着陆。obs: [x, y, vx, vy, angle, ω, left_contact, right_contact]。

### 1. Agent 发生了什么？
- 得分从 124→146，持续改善中
- episode_length = 1000（拉满），说明 agent **没有 crash，也没有快速完成任务**，而是在目标附近徘徊
- approach_quality_reward 每回合贡献 54.4，progress 仅 1.2 —— agent 学会了"靠近目标 + 低速 + 平稳"，但**没有动力去完成最后的着陆动作**（让支撑腿接触平台）

### 2. 哪个组件是主因？
- `approach_quality_reward` ratio=40:1，是主导信号。它鼓励「靠近、慢速、水平」，但不区分「悬停在上方」和「实际着陆」。这是典型的 **goal_near_oscillation** 模式。
- `attitude_penalty` ratio=-0.22，微弱背景约束，不是问题。
- 缺失信号：没有任何组件奖励着陆腿接触（obs[6]/obs[7]），导致 agent 在目标区悬停到 episode 结束。

### 3. 上轮改了什么？
- Iter 2→3：w_approach 从 0.3 降到 0.15，得分 +22。削弱 approach_quality 的方向有效，但单靠继续削弱不能解决"不落地"问题——需要新的落地激励信号。

### 修改方案
**层次 2：新增组件** —— 添加 `contact_bonus`：用 `prox_factor` 门控的着陆腿接触奖励。`prox_factor` 保证只有靠近目标时接触才有意义（远处意外接触不给奖励），连续形式提供梯度。系数 0.05，单腿接触时 ≈0.05，双腿 ≈0.10，约为 approach_quality 最大值的 2/3，足够形成落地激励但不至于主导整个信号。

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

    # 4. Contact bonus: reward leg contact gated by proximity to target.
    #    prox_factor gate ensures contact only matters near the landing site;
    #    continuous gradient guides the agent to descend the last distance.
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    contact_bonus = prox_factor * (left_contact + right_contact) * 0.05

    w_approach = 0.15  # unchanged from iter 3
    approach_quality_reward = w_approach * approach_quality
    total_reward = progress + approach_quality_reward + attitude_penalty + contact_bonus

    components = {
        "progress_reward": progress,
        "approach_quality_reward": approach_quality_reward,
        "attitude_penalty": attitude_penalty,
        "contact_bonus": contact_bonus,
    }

    return float(total_reward), components
```
