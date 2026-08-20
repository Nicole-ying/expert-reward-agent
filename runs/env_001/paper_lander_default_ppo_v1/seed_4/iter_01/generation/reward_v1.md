# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x, y, vx, vy, angle, angvel, left_contact, right_contact = obs

    # ----------------------------------------------------------------
    # Core mandatory roles
    # ----------------------------------------------------------------
    # 1. Goal proximity: distance to target pad at (0,0)
    distance = (x**2 + y**2)**0.5
    proximity_reward = -distance                     # main driving signal

    # 2. Orientation penalty: keep body upright and stable
    orientation_penalty = -0.1 * (angle**2) - 0.1 * (angvel**2)

    # ----------------------------------------------------------------
    # Soft landing & settling (mandatory role)
    # ----------------------------------------------------------------
    # Gating factor: strong influence only when close to the target
    proximity_gate = 1.0 / (1.0 + 5.0 * distance)

    # Speed penalty: punish high velocities when near the pad
    speed_sq = vx**2 + vy**2
    speed_penalty_gated = -0.5 * speed_sq * proximity_gate

    # ----------------------------------------------------------------
    # Conditional role: safe contact bonus
    # ----------------------------------------------------------------
    # Reward stable two‑leg contact when close to the pad
    contact_bonus = 0.5 * left_contact * right_contact * proximity_gate

    # ----------------------------------------------------------------
    # Combine components
    # ----------------------------------------------------------------
    total_reward = (
        proximity_reward +
        orientation_penalty +
        speed_penalty_gated +
        contact_bonus
    )

    components = {
        "proximity_reward": proximity_reward,
        "orientation_penalty": orientation_penalty,
        "speed_penalty_gated": speed_penalty_gated,
        "contact_bonus": contact_bonus
    }

    return float(total_reward), components
```

# reward_v1 设计说明

- **任务画像**：`navigation_goal_reaching` / `goal_approach_and_soft_contact`，主体需要在离散动作下到达中心平台并稳定停靠，同时保持直立姿态。

- **选用的奖励职责（从 mandatory/conditional 中选取）**：
  - `goal_proximity`：主学习信号，使用 `-distance` 驱使 agent 持续靠近原点。
  - `soft_landing_and_settling`：
    - 速度惩罚通过 gate 与距离耦合，只在平台附近施加，防止高速撞击或反弹。
    - 接触奖励作为“任务完成近似信号”，奖励双腿同时触地且位置靠近中心。
  - `orientation_penalty`：轻量二次惩罚防止倾覆，不影响正常姿态调整。

- **未被使用的职责**：
  - `engine_usage_penalty`：v1 阶段未加入，避免过早压制探索动作和必要的推力调整；能耗优化留到后续迭代。
  - `time_penalty`：环境无自然时间信号，且“尽快”已隐含于 proximity shaping，无需额外惩罚。
  - `progress_export_constant`：info 为空，无可信 success/failure 标志，无法使用。

- **为何不使用 terminal_success_reward / terminal_failure_penalty**：  
  环境未提供显式的成功/失败 flag（`explicit_success_flag_available = false`），info 为 `{}`，强制使用会引入不可靠信号，违反对环境卡片的承诺。

- **公式算子来源**：
  - `proximity_reward`：直接取负距离（dense_state_signal 的惩罚形式）。
  - `orientation_penalty`：二次惩罚（quadratic_penalty）。
  - `speed_penalty_gated`：gate 函数（bounded_signal / soft_health_gate）与速度平方结合，实现条件惩罚。
  - `contact_bonus`：利用 `joint_condition_proxy` 思想，将接触、位置通过 gate 相乘，构成软着陆正奖励。

- **组件数量控制**：共 4 个组件，覆盖 3 个 mandatory role 和 1 个条件 contact bonus，符合 v1 的 2–4 组件预算。

- **训练后应重点观察的 failure modes**：
  - 速度惩罚 gate 可能过弱，导致主体仍以高速撞击后弹飞；
  - contact_bonus 在非平台表面被意外触发（环境卡片未声明其他可接触物，风险低）；
  - orientation_penalty 权重过大可能阻碍姿态调整机动；
  - 缺少引擎惩罚可能使 agent 依赖连续点火维持悬停而不降落；
  - proximity 引导可能导致从侧面逼近后滑出视野。

- **后续迭代方向**：  
  - 若策略过于依赖主引擎悬停，可加入轻量 `engine_usage_penalty` 或动作变化惩罚；
  - 若稳定停靠不足，可增强速度惩罚的灵敏度或引入更精细的加速度约束；
  - 若早期 crash 频繁，可考虑增加对非平台接触的惩罚（需额外地面标志，当前不可用）。