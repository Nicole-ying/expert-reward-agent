# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- extract signals ---
    x_obs, y_obs = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- helper: distance to target pad centre ---
    dist_obs = (x_obs ** 2 + y_obs ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5

    # === 1. progress reward (improvement_delta) ===
    progress = dist_obs - dist_next   # positive when distance shrinks
    w_progress = 50.0
    progress_reward = w_progress * progress

    # === 2. attitude stabilisation (quadratic penalty) ===
    w_angle = 1.0
    w_angvel = 0.5
    attitude_penalty = -w_angle * (abs(angle_next) ** 2) - w_angvel * (angvel_next ** 2)

    # === 3. fuel / engine usage penalty ===
    if action == 0:
        fuel_cost = 0.0
    elif action == 2:   # main engine
        fuel_cost = 0.2
    else:               # left or right orientation engine
        fuel_cost = 0.1
    fuel_reward = -fuel_cost

    # === 4. soft landing component: bonus for dual contact near centre, penalty for high speed ===
    dist_threshold = 0.2
    proximity = max(0.0, 1.0 - dist_next / dist_threshold)   # 1 when dist=0, 0 when dist >= threshold
    dual_contact = left_contact * right_contact               # 1 if both legs touch, else 0
    velocity_mag = (vx_next ** 2 + vy_next ** 2) ** 0.5
    v_safe = 0.5   # maximum allowable landing speed for full bonus; beyond this gives penalty
    speed_factor = 1.0 - velocity_mag / v_safe                # can be negative if speed > v_safe
    w_landing = 40.0
    landing_reward = w_landing * dual_contact * proximity * speed_factor

    # --- total reward ---
    total_reward = progress_reward + attitude_penalty + fuel_reward + landing_reward

    components = {
        "progress": progress_reward,
        "attitude": attitude_penalty,
        "fuel": fuel_reward,
        "landing": landing_reward,
    }

    return float(total_reward), components
```

# reward_v1 设计说明

- **任务画像与动力学**：navigation_goal_reaching → goal_approach_and_soft_contact；离散推力控制，双支撑腿着陆器。
- **选用的奖励职责**（从 environment_card.md 的 mandatory / conditional 中按可用信号筛选）：
  - `approach_target_position` （主进程信号） → 使用 **improvement_delta** 算子，直接奖励每步距离缩小。
  - `stabilize_orientation_and_angular_velocity` （健康约束） → 使用 **quadratic_penalty** 对倾斜角和角速度施加轻量惩罚。
  - `penalize_engine_usage` （燃料效率） → 动作代价，主发动机代价略高于姿态发动机。
  - `dual_leg_contact_bonus` + `soft_landing_velocity_penalty` → 合并为一个 **joint_condition_proxy × bounded 速度因子**，仅在双腿同时接触且靠近平台中心时生效，速度越小奖励越高，速度超过安全阈值则变为惩罚。
- **排除的职责**：
  - `early_termination_success_bonus`：缺少明确的 success flag，且 `body_not_awake_or_settled` 可能在错误位置触发，禁止使用。
  - `terminal_success_reward` / `terminal_failure_penalty`：环境无显式 success / failure 信号，且 info 为空，无法可靠实现。
- **为什么没有使用 `terminal_success_reward`**：`explicit_success_flag_available=false`，不允许伪造，且任何基于 episode 终止原因的奖励都会因无可靠观测而引入噪声。
- **组件预算**：使用 4 个组件（progress、attitude、fuel、landing），符合 v1 推荐的上限，每个组件都有明确的数学形式和信号来源。
- **与历史尝试的区别**：放弃直接的状态负距离惩罚，改用每步 improvement delta 作为主进程，使 agent 在不移动时得分为零而非持续被罚，整体奖励有可能转为正值；速度惩罚被嵌入着陆组件中，且只在接触+接近中心时生效，避免了远距离时对正常运动速度的误伤。
- **后续迭代方向**：若 agent 仍出现高速撞击，可考虑将速度边界提示更早地加入（例如用 distance‑gated 垂直速度惩罚）；若燃料代价压制探索，可改为课程式权重；姿态稳定可升级为 soft_health_gate。
