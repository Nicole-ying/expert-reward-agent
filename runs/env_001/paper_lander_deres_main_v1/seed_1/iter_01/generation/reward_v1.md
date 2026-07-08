# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # === Progress reward (main learning signal) ===
    x_prev, y_prev = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]

    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    progress_reward = dist_prev - dist_next  # positive when getting closer

    # === Landing quality bonus (soft proxy for successful landing) ===
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    both_contact = (left_contact > 0.5) and (right_contact > 0.5)

    landing_quality_bonus = 0.0
    if both_contact:
        # position error from center of the pad
        pos_err = dist_next
        # velocity magnitude
        vx, vy = next_obs[2], next_obs[3]
        vel = (vx ** 2 + vy ** 2) ** 0.5
        # absolute body angle
        angle = abs(next_obs[4])

        # exponential-based quality score (max ~1.0 for perfect landing)
        # temperatures control how quickly the score decays with imperfection
        temp_pos = 0.2
        temp_vel = 0.5
        temp_angle = 0.1
        quality = (2.718281828 ** (-pos_err / temp_pos)) * \
                  (2.718281828 ** (-vel / temp_vel)) * \
                  (2.718281828 ** (-angle / temp_angle))
        landing_quality_bonus = quality

    total_reward = progress_reward + landing_quality_bonus

    components = {
        "progress_reward": progress_reward,
        "landing_quality_bonus": landing_quality_bonus
    }
    return float(total_reward), components
```

# reward_v1 设计说明

- **使用了哪些奖励组件**  
  - `progress_reward`（主学习信号）：基于当前状态与目标垫的欧氏距离变化，`dist_prev - dist_next`，稠密地引导飞行器靠近中央目标垫。  
  - `landing_quality_bonus`（任务完成近似信号）：仅当双脚触地时激活，由位置误差、速度大小、机体倾斜角的指数乘积组成，惩罚猛烈撞击或歪斜着地，鼓励轻柔、竖直、正中的着陆。

- **每个组件的角色**  
  - `progress_reward` 是 **唯一的主学习信号**（必须命名为 progress_reward），每一步都提供可微梯度，驱动飞行器向目标移动。  
  - `landing_quality_bonus` 是 **0~1 个任务完成近似信号**，在双脚着地时给予形状良好的奖励峰，帮助 agent 理解“怎样才算成功着陆”，弥补无显式成功标志的缺陷。即使触发率较低，其数值幅度自然较大，符合事件型奖励的特点。

- **为什么没有使用 terminal_success_re