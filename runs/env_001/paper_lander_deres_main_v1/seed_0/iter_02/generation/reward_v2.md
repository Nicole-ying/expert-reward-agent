## 诊断

### 1. Agent 发生了什么？
**Crash。** 100% 提前终止（20/20 集），episode_length 仅 68.5 步，score=-107。`original_env_reward` 每步 -1.55（ratio=-91），说明 agent 大量使用引擎且不断撞击——它在失控坠落，而不是受控着陆。

### 2. 哪个组件是主因？
**`landing_quality_reward` 几乎不触发（nonzero_rate=0.84%）**。原因：`contact_term = left_contact * right_contact` 要求双脚同时接地——这个条件在飞行过程中几乎永远为 0，甚至 crash 时也很少同时触发。结果：agent 唯一能收到的学习信号只有 `progress`（"靠近目标"），完全没有"减速"和"摆正姿态"的引导。它被教会冲向目标，但没被教会怎么停下来——于是直接撞上去。

### 3. 上一轮做了什么？
这是第一轮（iter=1），无历史改动需要避免。

### 修复方案
**层次 2：改数学形式。** 将 `landing_quality_reward` 从「二值接触门控 × 连续因子」改为「纯连续乘积 shaping」（bounded_continuous_proxy 技法）。去掉 `left_contact * right_contact` 这个杀死 99% 信号的二元门，用三个每步都触发的 bounded 因子（proximity × speed × angle）的乘积提供密集梯度，教会 agent 同时满足"靠近、减速、扶正"三个条件。权重随之调整（形态变了，系数自然变）。

只改这一个组件。

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

    # 2. Continuous approach quality shaping (replaces binary-contact-gated landing_quality)
    #    Three bounded factors [0,1] that fire EVERY step, providing dense gradient:
    #    - proximity: higher when closer to (0,0)
    #    - speed:     higher when slower
    #    - angle:     higher when more upright
    #    Product ensures all three must be satisfied simultaneously for high reward.
    prox_factor = 1.0 / (1.0 + 5.0 * dist_next)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    speed_factor = 1.0 / (1.0 + speed)
    angle_factor = 1.0 / (1.0 + 5.0 * abs(next_angle))
    approach_quality = prox_factor * speed_factor * angle_factor  # [0, 1], dense

    # 3. Orientation penalty (gentle, keeps the craft upright)
    attitude_penalty = -0.01 * abs(next_angle)

    # Combine
    w_approach = 0.3  # adjusted for denser signal (old w_landing=0.2 was for sparse binary-gated signal)
    total_reward = progress + w_approach * approach_quality + attitude_penalty

    components = {
        "progress_reward": progress,
        "approach_quality_reward": approach_quality,
        "attitude_penalty": attitude_penalty,
    }

    return float(total_reward), components
```