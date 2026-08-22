# Response Record

根据训练反馈，agent 得分严重为负且全部 episode 提前终止，说明当前奖励信号完全失效。根本问题在于 step_penalty 和 vel_penalty 构成了主要的负向驱动，迫使 agent 尽快结束 episode，而正向引导（距离减少、接触奖励）过弱且稀疏，导致无法学习到有意义的着陆行为。需要重新设计奖励框架：以“向目标移动”为核心的正向势场塑形，辅以触地阶段的软着陆惩罚和稳定着陆奖励，同时大幅降低生存惩罚的权重，让 agent 有足够的正向激励去完成下降和着陆。

由此，设计新奖励：
- **dist_reward**：作为主正向信号，用距离势能差引导 agent 向目标点 (0,0) 移动，系数提高到 8.0，确保 magnitude 占据主导。
- **vel_penalty**：仅在靠近地面（高度 <1.0）时惩罚高速，代替原来全时段的惩罚，鼓励软着陆。
- **angle_penalty**：轻度惩罚角度偏差，使姿态保持垂直。
- **contact_bonus**：增加部分接触奖励（单腿触地给 0.5），并保留稳定着陆（双腿触地且低速、小角度）的大额奖励 10.0，降低稀疏性。
- **fuel_penalty** 与 **step_penalty** 保持极小值，不再成为主要驱动力。

新奖励函数如下：

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y, vx, vy, angle, omega, lc, rc = obs
    nx, ny, nvx, nvy, nangle, nomega, nlc, nrc = next_obs

    # 1. 主正向：距离势场塑形，向目标 (0,0) 移动
    d_prev = (x * x + y * y) ** 0.5
    d_curr = (nx * nx + ny * ny) ** 0.5
    dist_reward = 8.0 * (d_prev - d_curr)

    # 2. 软着陆速度惩罚（仅在地面附近生效）
    ground_factor = 1.0 if ny < 1.0 else 0.0
    vel_penalty = -0.4 * (abs(nvx) + abs(nvy)) * ground_factor

    # 3. 角度偏离惩罚
    angle_penalty = -0.3 * abs(nangle)

    # 4. 接触奖励（降低稀疏性）
    if nlc > 0 and nrc > 0 and ny < 0.3 and abs(nangle) < 0.2 and abs(nvx) < 0.2 and abs(nvy) < 0.2:
        contact_bonus = 10.0
    elif nlc > 0 or nrc > 0:
        contact_bonus = 0.5
    else:
        contact_bonus = 0.0

    # 5. 燃油消耗惩罚
    fuel_penalty = -0.01 if action != 0 else 0.0

    # 6. 极轻生存步数惩罚
    step_penalty = -0.02

    total = (
        dist_reward
        + vel_penalty
        + angle_penalty
        + contact_bonus
        + fuel_penalty
        + step_penalty
    )

    components = {
        "dist_reward": dist_reward,
        "vel_penalty": vel_penalty,
        "angle_penalty": angle_penalty,
        "contact_bonus": contact_bonus,
        "fuel_penalty": fuel_penalty,
        "step_penalty": step_penalty,
    }
    return float(total), components
```
