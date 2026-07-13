# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # weights and sensitivities
    w_progress = 1.0
    w_angle_penalty = 0.2
    w_landing = 1.0
    a_v = 10.0       # sensitivity for vertical speed
    b_angle = 10.0   # sensitivity for body angle in landing quality

    # current and next distances to target (0,0)
    dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 1. Main progress signal: improvement towards target
    progress = w_progress * (dist - next_dist)

    # 2. Orientation stability: penalty for body angle to prevent tumbling and crash
    orientation_penalty = -w_angle_penalty * (next_obs[4] ** 2)

    # 3. Landing quality bonus: soft success proxy when contacts are active
    contact = max(next_obs[6], next_obs[7])  # any leg contact
    vertical_speed_factor = 1.0 / (1.0 + a_v * (next_obs[3] ** 2))
    angle_factor = 1.0 / (1.0 + b_angle * (next_obs[4] ** 2))
    landing_quality = w_landing * contact * vertical_speed_factor * angle_factor

    total_reward = progress + orientation_penalty + landing_quality

    components = {
        "progress": progress,
        "orientation_penalty": orientation_penalty,
        "landing_quality": landing_quality
    }

    return float(total_reward), components
```

# reward_v1 设计说明

- **任务类型**：`navigation_goal_reaching`，动力学子类型 `goal_approach_and_soft_contact`（2D 着陆器软着陆）。
- **设计依据**：以 `environment_card.md` 的 `reward_role_decomposition` 和可用信号为核心，遵循“主学习信号 + 约束 + 任务完成近似信号”的组件预算。
- **选择的奖励角色与公式映射**：
  - **主学习信号**：`progress_towards_target`（mandatory role），使用 `improvement_delta` 形式，奖励向目标平台移动的每一步距离改善（`dist - next_dist`）。
  - **稳定/安全约束**：`orientation_stability`（mandatory role）与 `crash_and_out_of_bounds_prevention` 合并，使用 `quadratic_penalty` 惩罚下一步的身体角度平方，抑制翻滚和碰撞倾向。
  - **任务完成近似信号**：`goal_landing_success` 和 `soft_landing_condition` 合并为 `landing_quality`，采用 `joint_condition_proxy` 风格：当任一条支撑腿接触时，以连续 bounded 函数鼓励低垂直速度和低角度，提供稠密的软着陆引导。
- **排除的角色及原因**：
  - `fuel_efficiency`（conditional）：节约燃料的引导会干扰必需的引擎使用，v1 阶段暂不加入。
  - `time_optimization`（avoid）：缺乏时间步数信号，强制时间优化无依据。
  - `exact_center_landing`（avoid）：过度约束中心对准会妨碍探索，且非任务必需。
- **不可用终端信号的说明**：
  - `explicit_success_flag_available=false`，因此不使用 `terminal_success_reward`。
  - `explicit_failure_flag_available=false`，因此不使用 `terminal_failure_penalty`。
- **留待后续迭代的职责**：燃料效率（`fuel_efficiency`）、动作平滑性代价、动态课程权重。
- **预期观察的失败模式**：着陆器可能过早关闭主引擎导致悬停失败；或为追求 progress 奖励高速俯冲导致硬碰撞；应通过 `landing_quality` 约束和终端物理检测逐步修正。
