# Response Record

# 设计理由

**修改了什么组件**：将 `landing_gate`（乘性门控因子）替换为 `landing_approach_reward`（加性密集奖励）。原设计中 `landing_gate` 乘到 `progress` 上，使得 progress 的信号被严重压制，尤其在 agent 状态较差时 gate 接近 0，progress 的梯度几乎消失；同时当 progress 为负时，gate 放大负贡献，导致 reward 剧烈波动。新组件变成一个独立的、始终非负的密集奖励，鼓励 agent 靠近原点并且保持小速度、小角度、小角速度，形成一条连续的“着陆走廊”梯度。

**数学形式**：  
`landing_approach_reward = proximity * safe_vy * safe_vx * safe_angle * safe_angvel`  
- `proximity = exp(-dist_next² / 1.0²)`，使得只有靠近目标区域才能获得显著奖励，避免在远处奖励“停滞”。  
- 每个 `safe_*` 因子是高斯函数，鼓励相应变量趋近 0。  
- 乘积形式保留，因为 landing 是一个联合条件，但不再作为 gate 乘到 progress 上，而是独立的正信号。  

**系数校准**：系数为 1.0。在典型“接近且姿态良好”状态（dist ≈ 0.3，速度约 0.1‑0.2）下，乘积 ≈ 0.02‑0.05，折算 per‑step ≈ 0.03，与 progress 的 per‑step 约 0.09 相近，不会支配 reward；同时 200 倍的接触成功奖励依然保持其“最大回报”地位。

**与已尝试历史的区别**：第 2 轮 best（‑18.80）使用了类似 `landing_approach_reward` 的密集接近奖励，但后来被稀疏的 `contact_success_reward` + gate 替换导致效果崩塌。本修改恢复了“密集着陆引导”这一关键信号，同时保留接触成功的高额奖励提供终极目标，避免重复第 3‑4 轮的失败。

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

    # 1. Progress: positive when moving closer to origin (centre of platform)
    dist_curr = (x_curr**2 + y_curr**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = 10.0 * (dist_curr - dist_next)

    # 2. Dense landing-approach reward — encourages proximity & safe landing states
    sigma_dist = 1.0
    proximity = 2.718281828 ** (- (dist_next**2) / (sigma_dist**2))

    sigma_vy = 0.5
    sigma_vx = 0.4
    sigma_angle = 0.2
    sigma_angvel = 0.4

    safe_vy = 2.718281828 ** (- (y_vel_next**2) / (sigma_vy**2))
    safe_vx = 2.718281828 ** (- (x_vel_next**2) / (sigma_vx**2))
    safe_angle = 2.718281828 ** (- (angle_next**2) / (sigma_angle**2))
    safe_angvel = 2.718281828 ** (- (ang_vel_next**2) / (sigma_angvel**2))

    landing_approach_reward = proximity * safe_vy * safe_vx * safe_angle * safe_angvel

    # 3. Sparse terminal success reward
    contact_flag = min(left_contact, right_contact)  # 1.0 only if both legs touch

    sigma_vy_success = 0.2
    sigma_vx_success = 0.2
    sigma_angle_success = 0.1
    sigma_angvel_success = 0.2

    safe_vy_success = 2.718281828 ** (- (y_vel_next**2) / (sigma_vy_success**2))
    safe_vx_success = 2.718281828 ** (- (x_vel_next**2) / (sigma_vx_success**2))
    safe_angle_success = 2.718281828 ** (- (angle_next**2) / (sigma_angle_success**2))
    safe_angvel_success = 2.718281828 ** (- (ang_vel_next**2) / (sigma_angvel_success**2))

    contact_success_reward = 200.0 * contact_flag * safe_vy_success * safe_vx_success * safe_angle_success * safe_angvel_success

    total_reward = progress + landing_approach_reward + contact_success_reward

    components = {
        'progress': progress,
        'landing_approach_reward': landing_approach_reward,
        'contact_success_reward': contact_success_reward
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 所有观测维度均已使用，信号覆盖完整；问题出在信号组合与校准——gate 压制 progress 导致梯度消失，并非信号缺失。
- **behavior**: agent 快速坠毁（len≈131），无法学到减速与姿态控制，因为 progress 被 gate 压制、正向引导过于稀疏。
- **signal**: 缺少密集的“着陆接近”引导；已有的 progress + gate 结构产生微弱甚至负向的梯度，contact_success 触发率仅 0.6%，无法提供连续学习信号。
- **level**: Level 2（将 gate 结构性变换为加性密集奖励）
- **hypothesis**: 恢复密集的接近‑安全奖励（landing_approach_reward）将为 agent 提供连续梯度，引导其靠近目标并降低速度/角度，从而提升存活时间，接触成功奖励随后可被有效触发，最终改善原始任务得分。
- **risk**: 若 proximity 的衰减不够快，agent 可能在远处学会“保持平稳但不移动”；当前 sigma_dist=1.0 使奖励主要集中在原点附近，风险可控。系数 1.0 带来的 per‑step 奖励量级适中，不会掩盖 progress 或接触成功信号。
