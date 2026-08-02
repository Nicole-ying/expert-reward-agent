# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Progress reward (improvement_delta on distance to origin)
    dist_curr = (x_curr**2 + y_curr**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = 10.0 * (dist_curr - dist_next)  # positive when getting closer

    # 2. Soft landing constraints
    k = 0.5  # desired vertical speed proportional to height
    desired_y_vel = -k * y_next
    vertical_error = y_vel_next - desired_y_vel
    penalty_y_vel = vertical_error**2

    penalty_x_vel = x_vel_next**2

    # Hinge penalties for angle and angular velocity
    angle_error = max(0.0, abs(angle_next) - 0.2)
    penalty_angle = angle_error**2

    ang_vel_error = max(0.0, abs(ang_vel_next) - 0.5)
    penalty_ang_vel = ang_vel_error**2

    w_y_vel = 1.0
    w_x_vel = 1.0
    w_angle = 2.0
    w_ang_vel = 0.5

    soft_landing_penalty = (w_y_vel * penalty_y_vel +
                            w_x_vel * penalty_x_vel +
                            w_angle * penalty_angle +
                            w_ang_vel * penalty_ang_vel)

    # 3. Landing bonus (joint condition proxy)
    landing_bonus = 0.0
    if left_contact == 1.0 and right_contact == 1.0:
        speed = (x_vel_next**2 + y_vel_next**2) ** 0.5
        # Soft bonus: higher for low speed and upright
        landing_bonus = 20.0 / (1.0 + speed) * (1.0 / (1.0 + abs(angle_next)))

    total_reward = progress - soft_landing_penalty + landing_bonus

    components = {
        'progress': progress,
        'soft_landing_penalty': soft_landing_penalty,
        'landing_bonus': landing_bonus
    }

    return float(total_reward), components
```

# reward_v1 设计说明

- **任务家系**：`navigation_goal_reaching`，子类型 `goal_approach_and_soft_contact`。
- **选用奖励角色**：
  - **主学习信号**：`progress`（使用 `improvement_delta` 算子，基于水平/垂直距离变化）。每步都有梯度，引导飞行器朝目标原点靠近。
  - **稳定/安全约束**：`soft_landing_penalty`，结合垂直速度误差（二次惩罚）、水平速度（二次惩罚）、姿态角（hinge二次）和角速度（hinge二次）。防止粗暴着陆、过度倾斜或震荡。
  - **任务完成近似信号**：`landing_bonus`（仅双腿同时接触时生效），基于速度和姿态的连续因子乘积，提供着陆阶段的正向软目标。
- **被排除的角色**：
  - `terminal_success_reward` 和 `terminal_failure_penalty` – 环境未提供显式成功/失败标志，因此未使用硬编码的终止奖励。
  - `action_efficiency` – 在离散动作空间中效率代价默认留到后续迭代。
  - `soft_health_gate` 和 `preview_conditioned_reward` – v1 初期不加入复杂门控。
- **已知限制**：着陆奖励仅当双腿同时接触时出现，前期梯度依赖 progress 和约束；训练后应观察水平振荡、垂直弹跳或角度过冲等行为，必要时再调整 hinge 阈值或引入门控。