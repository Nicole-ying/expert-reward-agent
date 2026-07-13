# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract current and next observations
    x = obs[0]
    y = obs[1]
    # x_vel, y_vel not directly used from obs, use next_obs for state‑transition reward
    # body_angle, angular_vel from obs are not needed, we use next_obs

    nx = next_obs[0]
    ny = next_obs[1]
    nx_vel = next_obs[2]
    ny_vel = next_obs[3]
    nbody_angle = next_obs[4]
    nangular_vel = next_obs[5]
    nleft_contact = next_obs[6]
    nright_contact = next_obs[7]

    # Distance to target (origin) before and after the step
    dist_curr = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5

    # Component A: approach progress (distance reduction)
    progress = max(0.0, dist_curr - dist_next)

    # Component B: velocity penalty (continuous, on next state)
    velocity_penalty = -(nx_vel**2 + ny_vel**2)

    # Component C: angular velocity penalty (continuous, on next state)
    angular_penalty = -(nangular_vel**2)

    # Component D: soft landing quality (joint‑condition proxy)
    both_legs = nleft_contact * nright_contact          # 1.0 only when both legs contact
    speed_factor = max(0.0, 1.0 - abs(nx_vel) - abs(ny_vel))
    angle_factor = max(0.0, 1.0 - abs(nbody_angle))
    angvel_factor = max(0.0, 1.0 - abs(nangular_vel))
    landing_quality = both_legs * speed_factor * angle_factor * angvel_factor

    # Weights (tuned to make approach dominant while softly limiting velocity and oscillation)
    w_progress = 1.0
    w_vel = 0.05
    w_ang = 0.01
    w_land = 2.0

    reward_progress = w_progress * progress
    reward_vel = w_vel * velocity_penalty
    reward_ang = w_ang * angular_penalty
    reward_land = w_land * landing_quality

    total_reward = reward_progress + reward_vel + reward_ang + reward_land

    components = {
        'progress': reward_progress,
        'velocity_penalty': reward_vel,
        'angular_penalty': reward_ang,
        'landing_quality': reward_land
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 任务画像与职责选择
- **任务族群**: `navigation_goal_reaching`
- **动力学子类**: `goal_approach_and_soft_contact`
- **核心职责**:
  - `approach_and_landing`：引导飞行器向目标着陆垫移动并安全减速
  - `soft_contact_stabilization`：确保双腿同时平稳触地，姿态和速度符合着陆要求

## 信号映射与算子选择
| 角色 | 使用信号 | 公式算子 | 理由 |
|---|---|---|---|
| `approach_and_landing` | `x_position, y_position` | `improvement_delta`（`max(0, dist_curr - dist_next)`） | 每步给出距离减小的正奖励，不惩罚暂时远离，避免抑制探索 |
| `approach_and_landing` | `x_velocity, y_velocity` | `quadratic_penalty`（`-(vx²+vy²)`） | 软约束水平与垂直速度，防止猛烈撞击，给每步连续梯度 |
| `soft_contact_stabilization` | `angular_velocity` | `quadratic_penalty`（`-ω²`） | 轻量抑制姿态晃动，与速度惩罚一起构成安全约束 |
| `soft_contact_stabilization` | `left_contact, right_contact,` `x_velocity, y_velocity, body_angle, angular_velocity` | `joint_condition_proxy`（乘积形式的多个 `bounded` 因子） | 只在双腿接触时激活，奖励接近零速度、小姿态误差、低角速度的平稳着陆，作为接近成功的软信号 |

- **距离改善**（progress）是主学习信号，提供每步向目标靠近的正向驱动力。
- **速度/角速度惩罚**是连续安全约束，防止 agent 高速撞击或翻滚。
- **着陆质量**在接触时才产生奖励，不干扰空中探索；乘积形式确保所有条件均好时奖励最高，鼓励稳定着陆。

## 排除的角色及其原因
- `fuel_efficiency`：v1 阶段不引入动作惩罚，避免损害探索和任务成功率。
- `time_efficiency`：无独立时间信号，常数步惩罚在早期容易诱导崩溃（agent 选择快速坠毁以结束 episode），暂不使用。
- `precise_posture_alignment`：任务只要求大致稳定，精确角度约束会限制必要的姿态调整。
- `exploration_bonus`：距离信号已经提供稠密反馈，无需额外探索奖励。
- `terminal_success_reward` / `terminal_failure_penalty`：环境中无显式成功/失败标志（`explicit_success_flag_available=false`），info 为空，不能依赖。

## 留待后续迭代的职责
- **燃料效率**：当基础着陆成功率达到较高水平后，可逐步引入仅对主发动机或所有非零动作的惩罚。
- **时间效率**：若训练表现出缓慢悬停行为，可考虑极小的常数步惩罚，但需密切监控崩溃率。
- **动态权重或课程**：若早期靠重力下坠成功率低，可暂时降低速度惩罚权重，但 v1 先保持静态平衡。

## 训练后应观察的 failure modes
- **自由下坠（无发动机）**：距离减少大但速度也大，速度惩罚若过小则 agent 学会坠落；检查 `velocity_penalty` 分量的均值，必要时增大 `w_vel`。
- **悬停或水平漂移**：progress 接近于 0，agent 仅偶尔点推；观察 episode 长度是否异常增大，可考虑后期加入轻微常数惩罚。
- **单腿先触地翻滚**：landing_quality 可能提前激活少量奖励，导致过早追求接触；监控 `landing_quality` 的触发步及其与 crash 的相关性，若问题严重可提高 `speed_factor` 阈值或暂时去掉乘积中的 `angle_factor` 放宽要求。
