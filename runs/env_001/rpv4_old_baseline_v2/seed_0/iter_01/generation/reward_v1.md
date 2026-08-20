# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x = obs[0]          # horizontal distance to target pad
    y = obs[1]          # vertical distance (height) above pad
    x_vel = obs[2]      # horizontal velocity
    y_vel = obs[3]      # vertical velocity (positive upward, downward negative)
    angle = obs[4]      # body angle (pitch)
    ang_vel = obs[5]    # angular velocity (not used in v1, reserved)
    left_contact = obs[6]
    right_contact = obs[7]

    # 1. Goal proximity: bounded, always active, encourages bringing x and y close to zero
    #    Using individual bounded forms to avoid dominating entire reward
    kx = 0.5
    ky = 1.0
    prox_x = 1.0 / (1.0 + kx * abs(x))
    prox_y = 1.0 / (1.0 + ky * abs(y))
    proximity = prox_x + prox_y

    # 2. Soft landing reward: joint‑condition proxy, active only when near the pad
    threshold_y = 0.2
    landing_weight = 2.0
    if y < threshold_y:
        # Contact factor: both legs must touch for full 1.0, continuous gradient
        contact_factor = (left_contact + right_contact) / 2.0
        # Speed factor: low total speed is desired
        speed_factor = 1.0 / (1.0 + 5.0 * (abs(x_vel) + abs(y_vel)))
        # Angle factor: body should be nearly upright
        angle_factor = 1.0 / (1.0 + 10.0 * abs(angle))
        landing_bonus = landing_weight * contact_factor * speed_factor * angle_factor
        landing_reward = landing_bonus
    else:
        landing_reward = 0.0

    # 3. Fuel efficiency penalty: simple linear penalty for any engine use
    fuel_penalty = -0.05 if action != 0 else 0.0

    total_reward = proximity + landing_reward + fuel_penalty

    components = {
        "proximity": proximity,
        "landing_reward": landing_reward,
        "fuel_penalty": fuel_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

- **selected task_family / dynamics_subtype**  
  `navigation_goal_reaching` / `goal_approach_and_soft_contact`.  
  任务核心是驱使二维刚体到达并安全软着陆于目标垫，同时节省燃料。

- **selected reward roles**  
  依照 `reward_role_decomposition` 的 mandatory roles 选取了三个职责：  
  - `goal_proximity`  – 主学习信号，引导舰体向垫子靠近（x,y 距离）。  
  - `soft_landing`    – 任务完成近似，确保接近垫面时低速、双腿接触且姿态正立。  
  - `fuel_efficiency` – 每步对非空闲动作施加轻量惩罚，满足次要目标。  

  条件职责 `orientation_stabilization` 和 `speed_moderation` 未在 v1 中单独使用，因为 `soft_landing` 已通过速度/角度因子隐式携带了这些约束，避免过早限制机动性。

- **role_to_signal_mapping**  
  | Role | Usable signals | Used |
  |---|---|---|
  | goal_proximity | `x_position, y_position` | `obs[0], obs[1]` (distance) |
  | soft_landing   | `y_position, y_velocity, left/right_contact, body_angle` | `obs[1], obs[2], obs[3], obs[6], obs[7], obs[4]` |
  | fuel_efficiency| `action` | `action` (non‑zero) |

- **每个 role 选择的 formula operator**  
  - `goal_proximity`: 采用 `bounded_signal` 形式（`1/(1+k*|error|)`），避免距离极大时奖励无界，并为 x,y 分别提供梯度。  
  - `soft_landing`: 使用 `joint_condition_proxy`，通过连续因子乘积构造软成功条件，仅在 `y < 0.2` 时激活，避免稀疏奖励问题。  
  - `fuel_efficiency`: 使用简单的 `linear_penalty`（动作非零即扣分）。

- **excluded roles 及原因**  
  - `survival_balance`：环境不要求持续平衡，只有最终着陆稳定性，不必全局惩罚。  
  - `sparse_exploration_bonus`：观测连续且清晰，不需要额外探索职责。  
  - `orientation_stabilization`（条件职责）：留到后续迭代，当前由 `soft_landing` 的角度因子覆盖。  
  - `speed_moderation`：同样由 `soft_landing` 的速度因子处理，避免重复叠加。

- **为什么没有使用 terminal_success_reward / terminal_failure_penalty**  
  environment_card 明确声明 `explicit_success_flag_available: false` 和 `explicit_failure_flag_available: false`，而 `info` 为空。无可用标志。因此所有成功/失败信号必须从观测中构造，本设计用 `joint_condition_proxy` 代替。

- **哪些职责留到后续迭代**  
  - 条件姿态稳定约束（激活范围更精细，可能配合进度课程）。  
  - 动作平滑/功率积分（动作代价精细化）。  
  - 动态权重课程（training_progress 介入）。  
  - 接触顺序/两条腿依次着陆的时序引导（当前仅用总接触因子）。

- **训练后应观察哪些 failure modes**  
  - **自由落体/不启动主引擎**：平均 episode 短，`y_velocity` 持续负向，减速为零。需观察 `proximity` 是否提供足够驱动力，必要时提高垂直接近奖励。  
  - **假成功（停在垫外）**：终止时 `x` 或 `y` 离零较远，双腿未全触地。可通过调试 `soft_landing` 阈值及因子形状加固位置条件。  
  - **高速/倾斜撞击**：crash 终止，垂直速度大，角度大。检查 `soft_landing` 中速度/角度因子的灵敏度与 weight。  
  - **过度消耗燃料**：非零动作比例极高但成功着陆。可提高 `fuel_penalty` 系数或改用动作功率积分。  
  - **悬停振荡迟迟不降落**：高度振荡，`angle` 频繁校正。后续可加入渐进加速收敛项或减少接近阶段的方向发动机扰动。