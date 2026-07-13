# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # === Unpack next_obs ===
    x       = next_obs[0]
    y       = next_obs[1]   # height above landing pad, assumed non‑negative
    vx      = next_obs[2]
    vy      = next_obs[3]
    angle   = next_obs[4]
    angvel  = next_obs[5]
    left_c  = next_obs[6]   # 0 or 1
    right_c = next_obs[7]   # 0 or 1

    # === Useful derived quantities ===
    height = max(y, 0.0)              # protect against negative penetration
    dist_to_target = (x**2 + height**2) ** 0.5
    speed_sq = vx**2 + vy**2
    inv_height_scale = 1.0 / (0.1 + height)   # increases as height decreases

    # === Hyper‑parameters (v1) ===
    w_prox    = 0.2
    w_vel     = 0.05
    w_orient  = 0.02
    w_contact = 5.0

    temp_speed  = 0.2      # velocity penalty / safe‑contact temperature
    temp_angle  = 0.04     # angle temperature for safe contact
    temp_x      = 0.2      # horizontal distance temperature for safe contact

    # === 1. Target approach proximity (main learning signal) ===
    # Negative euclidean distance – every step gives gradient towards target
    proximity_reward = -w_prox * dist_to_target

    # === 2. Velocity norm penalty (safety constraint) ===
    # Quadratic penalty on speed, scaled by 1/(0.1+height) -> low altitude penalties are stronger
    velocity_penalty = -w_vel * speed_sq * inv_height_scale

    # === 3. Orientation stability (safety constraint) ===
    # Quadratic penalty on angle + angular velocity, also height‑scaled
    orientation_penalty = -w_orient * (angle**2 + angvel**2) * inv_height_scale

    # === 4. Safe contact encouragement (soft proxy for landing) ===
    # Only active when both support legs touch the pad.
    # Further modulated by near‑zero velocity, near‑upright angle, and near‑center horizontal position.
    safe_contact_bonus = 0.0
    if left_c == 1 and right_c == 1:
        # Use exponential factors because numpy is forbidden and we want smooth [0,1] scaling.
        e = 2.718281828
        speed_factor   = e ** (-speed_sq / temp_speed)
        angle_factor   = e ** (-angle**2 / temp_angle)
        x_prox_factor  = e ** (-x**2 / temp_x)

        safe_contact_bonus = w_contact * speed_factor * angle_factor * x_prox_factor

    # === Assemble total reward ===
    total_reward = proximity_reward + velocity_penalty + orientation_penalty + safe_contact_bonus

    components = {
        "proximity_reward":      proximity_reward,
        "velocity_penalty":      velocity_penalty,
        "orientation_penalty":   orientation_penalty,
        "safe_contact_bonus":    safe_contact_bonus
    }

    return float(total_reward), components
```

# reward_v1 设计说明

**selected task_family / dynamics_subtype**：  
`navigation_goal_reaching` – `goal_approach_and_soft_contact`，离散控制 2D 飞行器着陆到中央目标垫。

**selected reward roles（按 environment_card 的 `reward_role_decomposition` 强制要求）**：  
1. `target_approach_proximity` – 引导飞行器向目标垫中心 (x=0, y=0) 移动。  
2. `velocity_norm_penalty` – 惩罚过大速度，尤其在接近地面时，强制减速以避免撞击。  
3. `orientation_stability` – 惩罚大角度和高角速度，保证平稳、直立着陆。  
4. `safe_contact_encouragement` – 当双支撑腿同时触地且速度、角度、水平位置都良好时给予额外正奖励，弥补无显式成功标志的不足。

**role_to_signal_mapping 与 formula operator 选择**：  

| role | signals used | operator | 说明 |
|------|--------------|----------|------|
| target_approach_proximity | `x, y` | dense_state_signal (linear negative distance) | 每步提供梯度，方向指向中心 |
| velocity_norm_penalty | `vx, vy` + `height` | quadratic_penalty + conditional scaling | 速度平方罚项，用 `1/(0.1+height)` 放大低空惩罚 |
| orientation_stability | `angle, angvel` + `height` | quadratic_penalty + conditional scaling | 角度与角速度平方罚项，同样随高度缩放 |
| safe_contact_encouragement | `left_c, right_c, vx, vy, angle, x` | joint_condition_proxy (乘积指数因子) | 仅双接触时激活，奖励由速度、角度、水平偏差三个指数因子连乘 |

- **为什么用高度衰减？**  高空允许稍快的下降速度与一定的姿态偏斜，靠近垫面时必须严格减速和立正，否则 crash。阈值式 hinge 可能使 agent 仓促减速，故采用平滑缩放 `1/(0.1+height)`。  
- **safe_contact 为什么用指数因子？**  三个因子均连续、平滑，即使着陆条件不完全理想也给出小正信号，避免奖励完全塌缩，同时引导向最优着陆状态。

**excluded roles 及原因**：  
- `engine_usage_regularization`：节约燃料，v1 不干扰任务方向的学习，留到后续迭代。  
- `soft_time_pressure`：隐含时间压力，当前环境无明确步数限制，v1 先不加，避免干扰安全约束。  
- `boundary_safety_penalty`、`crash_specific_penalty`、`exact_landing_zone_incentive`：均在 environment_card 中列为 avoid_roles（信号缺失或已有替代）。

**为什么没有使用 terminal_success_reward / terminal_failure_penalty**：  
- `explicit_success_flag_available: false` 且 `info` 为空，无法获取任何终止原因标志。  
- 即使能从 `done` 推断，奖励
