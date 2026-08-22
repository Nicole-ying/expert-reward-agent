# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- unpack observations ---
    x_pos = obs[0]           # horizontal distance to target
    y_pos = obs[1]           # vertical distance to target
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    # --- hyperparameters ---
    w_dist = 0.5            # approach reward weight
    w_vel = 0.02            # velocity penalty weight
    w_angle = 0.1           # body angle penalty weight
    w_angvel = 0.05         # angular velocity penalty weight
    w_boundary = 0.5        # horizontal out-of-bounds penalty
    w_landing = 1.0         # landing quality bonus weight

    safe_x_bound = 1.0       # x beyond this is considered unsafe
    vel_thresh = 0.5         # velocity below which landing is considered stable
    angle_thresh = 0.2       # body angle below which landing is considered upright

    # 1. Approach reward (main learning signal)
    dist_sq = x_pos**2 + y_pos**2
    approach = -w_dist * dist_sq

    # 2. Stability penalties (soft constraints)
    vel_penalty = -w_vel * (x_vel**2 + y_vel**2)
    angle_penalty = -w_angle * body_angle**2 - w_angvel * ang_vel**2

    # 3. Horizontal boundary penalty (prevent drifting out of viewport)
    if abs(x_pos) > safe_x_bound:
        boundary_penalty = -w_boundary * (abs(x_pos) - safe_x_bound)**2
    else:
        boundary_penalty = 0.0

    # 4. Landing quality bonus (joint condition proxy)
    both_legs = left_contact * right_contact  # = 1.0 only if both legs touch
    # bounded stability factors
    vel_stable = max(0.0, 1.0 - abs(x_vel) / vel_thresh)
    yvel_stable = max(0.0, 1.0 - abs(y_vel) / vel_thresh)
    angle_stable = max(0.0, 1.0 - abs(body_angle) / angle_thresh)
    landing_quality = both_legs * vel_stable * yvel_stable * angle_stable
    landing_bonus = w_landing * landing_quality

    total_reward = approach + vel_penalty + angle_penalty + boundary_penalty + landing_bonus

    components = {
        "approach_reward": approach,
        "velocity_penalty": vel_penalty,
        "angle_penalty": angle_penalty,
        "boundary_penalty": boundary_penalty,
        "landing_bonus": landing_bonus
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 选定的任务画像与动力学子类
- **task_family**: `navigation_goal_reaching`
- **dynamics_subtype**: `goal_approach_and_soft_contact`（离散动作、带两支撑腿的刚体飞行器）

## 选定的奖励职责及信号映射
| role | formula operator | signals used | notes |
|---|---|---|---|
| `approach_target` (mandatory) | `dense_state_signal` (negative quadratic penalty) | `x_position`, `y_position` | 每步都驱动 agent 向目标缩短欧氏距离 |
| `soft_landing` (mandatory) | `joint_condition_proxy` + bounded factors | `left_contact`, `right_contact`, `x_velocity`, `y_velocity`, `body_angle` | 乘积形式奖励双腿着地、低速、直立的“着陆质量” |
| 稳定/安全约束 | `quadratic_penalty` + `hinge penalty` | `x_velocity`, `y_velocity`, `body_angle`, `angular_velocity`, `x_position` | 抑制高速移动、过大倾角、角速度以及超出水平视口 |

## 排除的职责及原因
- **`terminal_success_reward`**：`explicit_success_flag_available = false`，且无法获取 `terminated` 标志，不能只在终步奖励。由 `landing_bonus` 提供每步可微的软成功代理。
- **`terminal_failure_penalty`**：`explicit_failure_flag_available = false`，无法可靠判断失败瞬间；水平出界已由 `boundary_penalty` 连续惩罚，高速与倾角由惩罚项约束。
- **`fuel_penalty`**：v1 默认不加；节省燃料是次要目标，过早弱惩罚可能压制必要探索；若后续观察到无效频繁点火再迭代加入。
- **`progress_speed_encouragement`**：`approach_reward` 已提供密集的距离改进梯度，额外的时间步负奖励或 potential shaping 会与主信号重叠，且可能引入无步限环境下的风险。

## 为什么没有使用 terminal 事件
- 环境 `info` 为空，无法读取 `terminated` 标志或终止原因；
- 依靠每步观测计算软着陆代理可以持续提供梯度，优于在无法确定终步时强加一次性大奖励（极易产生奖励黑客）。

## 留到后续迭代的职责
- 燃料效率/动作代价（若观察到过度的引擎使用）；
- 动态课程权重（例如靠近地面时自动增强 `landing_bonus` 和速度惩罚，需引入 `training_progress` 或高度门控）；

## 训练后应重点观察的 failure modes
| 模式 | 检测方法 | 可能的补救 |
|---|---|---|
| 高空悬停不下降 | `y_pos` 长期为正值，速度接近零 | 加大 `approach_reward` 权重或引入垂直速度的轻量负塑造 |
| 高速俯冲撞击 | 终止前 `y_vel` 极大负值，双腿未接触 | 提高 `velocity_penalty` 权重，或将 `landing_bonus` 中的速度阈值减小 |
| 翻滚后坠毁 | `body_angle` 与 `ang_vel` 剧烈振荡 | 加大 `angle_penalty` 和 `w_angvel` |
| 水平漂移出界 | `abs(x_pos)` 持续增大超出安全范围 | 增大 `boundary_penalty` 或降低 `safe_x_bound` |
| 仅单腿触地后终止 | `landing_bonus = 0` 但某一接触腿为 0 | 乘积形式已要求双腿触地，模型应自然学会同时用双腿 |

凭借上述设计，v1 使用四个组件（主信号 + 两个轻量约束 + 一个软成功代理），符合职责预算，并为着陆任务提供稠密、可微的梯度。
