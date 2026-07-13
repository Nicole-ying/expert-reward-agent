# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # distance to target (center pad)
    dist = (obs[0]**2 + obs[1]**2)**0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2)**0.5

    # 1. primary learning signal: approach improvement (improvement_delta)
    progress = 1.0 * (dist - next_dist)

    # 2. stability constraint: body angle near zero (linear penalty)
    angle_penalty = -0.1 * abs(next_obs[4])

    # 3. safety constraint: low velocities (quadratic penalty on both x and y speed)
    speed_penalty = -0.1 * (next_obs[2]**2 + next_obs[3]**2)

    # 4. soft landing proxy: both legs in contact (continuous product)
    contact_reward = 0.2 * (next_obs[6] * next_obs[7])

    total_reward = progress + angle_penalty + speed_penalty + contact_reward
    components = {
        "progress": progress,
        "angle_penalty": angle_penalty,
        "speed_penalty": speed_penalty,
        "contact": contact_reward
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 任务画像与职责选择

- **task_family**：`navigation_goal_reaching`
- **dynamics_subtype**：`goal_approach_and_soft_contact`（2D 飞行器软着陆）
- **selected reward roles**
  - **主学习信号**：`progress`（接近目标）— mandatory，每步提供梯度，直接驱动“到达目标平台”。
  - **稳定/安全约束**：`angle_penalty`（身体水平）— 防止过度倾斜导致 crash。
  - **速度约束**：`speed_penalty`（低速度）— 着陆要求速度趋零，且避免高速撞击地面。
  - **任务完成近似信号**：`contact`（双腿接触）— 软着陆必要条件，连续化提供稀疏引导。

## 信号映射与公式算子

- `progress`：映射自 `obs[0], obs[1]`（位置）→ 到目标距离（`dist`）。使用 **improvement_delta**（`dist - next_dist`），靠近得正奖，远离受罚。每步稠密，无阈值风险。
- `angle_penalty`：映射自 `obs[4]`（身体角度）。使用 **dense_state_signal** 的线性 penalty（`-w * abs(next_obs[4])`），避免二次项对微小扰动过度敏感。
- `speed_penalty`：映射自 `obs[2], obs[3]`（线速度）。使用 **quadratic_penalty**（`-w * (vx² + vy²)`），轻量抑制高速，但低速时不压死探索。
- `contact`：映射自 `obs[6], obs[7]`（支撑腿 0/1 值）。采用连续乘积 `next_obs[6] * next_obs[7]`，近似 **joint_condition_proxy** 的简化形式，双腿同时接触给予最大奖励，无接触则为零。由于底层信号是二值（或接近二值），该乘积退化为稀疏引导，但能提供着陆阶段的额外梯度。

## 排除的角色与原因

- **terminal_success_reward** — 环境不提供显式 success flag（`explicit_success_flag_available: false`），无法安全使用。
- **terminal_failure_penalty** — 无显式 failure flag（crash/出界仅通过终止条件隐式体现，无 `info` 字段）。
- **效率/动作代价** — v1 优先学习正确着陆姿态与位置，燃料优化留到后续迭代。
- **soft_health_gate** — 当前 terminated 率未知，且主奖励 progress 与安全约束暂时独立，不必过早引入门控。
- **curriculum / dynamic weighting** — v1 无训练进度依赖，避免过早复杂化。

## 后续迭代预留

- 速度约束可升级为 **hinge** 形式（仅在超过安全阈值时惩罚），避免抑制正常飞行速度。
- 接触奖励可结合高度或距离门控（如 `contact * (1 - |y|/threshold)`），防止过早接触导致无意义的行为。
- 引入轻量时间/燃料惩罚（`-0.01` 每步或 `-0.05` 当 action≠0），促进效率。
- 若 terminated 率高且主信号在失败 episode 中仍为正，可考虑将 `angle_penalty` 或 `abs(vy)` 做成 `soft_health_gate` 乘到 progress 上。

## 预期需观察的 failure modes

- **速度过高着陆**：`speed_penalty` 权重偏弱可能导致 agent 为获得 `progress` 而高速冲向平台，触发 crash。
- **单腿接触后停滞**：`contact` 奖励可能使 agent 满足于仅部分接触，不再调整姿态使双腿同时着地。
- **角度抖动**：为同时满足 progress 和 angle_penalty 的权衡，可能出现连续小幅度旋转。
- **bouncing**：若着陆速度仍偏大，可能出现弹跳，需后续通过垂直速度约束或地面接触后的稳定性奖励解决。
