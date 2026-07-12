# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation signals
    body_z = obs[0]                     # body height
    vx = obs[13]                        # forward velocity (world x)
    quat_x = obs[2]
    quat_y = obs[3]

    # ---- 1. Survival gate: height factor ----
    # height health: allow full reward when body_z is in [0.4, 0.8]
    # linear ramp from 0 at boundary (0.2 or 1.0) to 1 at safe zone
    low_safe  = max(0.0, min(1.0, (body_z - 0.2) / 0.2))   # distance above 0.2
    high_safe = max(0.0, min(1.0, (1.0 - body_z) / 0.2))   # distance below 1.0
    height_factor = min(low_safe, high_safe)                 # clamp by the more critical side

    # ---- 2. Survival gate: uprightness factor ----
    # compute body_up_z from quaternion: how close the body's z-axis is to world z
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_factor = max(0.0, min(1.0, body_up_z))          # clip to [0,1], 1 = fully upright

    survival_gate = height_factor * upright_factor

    # ---- 3. Main progress component (gated) ----
    w_progress = 1.0
    forward_gated = w_progress * vx * survival_gate

    # ---- 4. Light energy efficiency term ----
    w_energy = 0.0005
    action_penalty = -w_energy * sum(a**2 for a in action)

    # Total reward
    total_reward = forward_gated + action_penalty
    components = {
        'forward_gated': forward_gated,
        'action_penalty': action_penalty
    }
    return float(total_reward), components
```

# reward_v1 设计说明

**selected task_family / dynamics_subtype**  
`locomotion_continuous_control` / `multi_legged_body_locomotion`：四足机器人通过连续扭矩控制实现稳定向前移动，主要目标是产生持续的前进速度，同时保持身体处于安全高度与姿态范围内。

**selected reward roles**  
- **forward_progress**（主学习信号）：直接奖励前进速度 `body_x_velocity`，告诉 agent “向前动就能得分”。  
- **survival_and_health**（必须稳定约束）：通过软门控 `soft_health_gate` 将身体高度和直立度融合进主奖励，避免 agent 在危险姿态下仍获得高额进度奖励。  
- **energy_efficiency**（可选标准角色）：极小的动作幅度惩罚，抑制无意义的过大扭矩，为后续步态效率优化留出接口。这符合 “动作维度 ≥ 6 时考虑轻量效率约束” 的原则。

**role_to_signal_mapping 与公式算子**  
- `forward_progress` → `body_x_velocity (obs[13])`，使用 `dense_state_signal` 的线性形式：`w * vx`。  
- `survival_and_health` → `body_z (obs[0])` 与旋转四元数计算得的 `body_up_z`。两个信号均被转化为连续 bounded 因子 (`height_factor`, `upright_factor`)，然后以乘积形式构成 `survival_gate`，再乘到主奖励上。这是一种 `soft_health_gate`（算子 2.6）的典型应用：安全时 gate≈1，接近危险边界时 smooth 衰减主奖励，而不引入独立的强惩罚项，避免过度压制探索。  
- `energy_efficiency` → `action` 8 维扭矩，使用 `quadratic_penalty`（`-w * sum(a²)`) 实现最小的能耗倾向。

**excluded roles 及原因**  
- `lateral_stability`、`body_orientation_penalty`：v1 不加入，因为早期引入侧向/角速度约束可能抑制运动多样性，留到后续迭代当 agent 已经能稳定前进后作为精调项。  
- `action_smoothing`：禁止。奖励函数无状态，无法获取前一帧动作，不能计算动作差分。  
- `contact_force_management`、`distance_progress_reward`：均因缺失必要信号（足端接触状态、全局 x 位置）而无法使用。  
- `official_reward_terms`：环境明确屏蔽且禁止使用。

**为什么没有 terminal_success_reward / terminal_failure_penalty**  
- 环境中没有显式成功标志（`explicit_success_flag_available=false`）。  
- 失败终止虽然存在（高度越界或数值异常），但没有独立的失败 label 传入 `info`，无法安全地设计专用 penalty。v1 转而使用连续 soft gate 在终止边界之前提供密集梯度，用衰减的主奖励替代事后惩罚。

**留到后续迭代的职责**  
- 侧向速度惩罚 (`body_y_velocity`)：当 agent 学会基本前进后可加入，抑制漂移。  
- 俯仰/滚转角速度约束：进一步细化身体稳定性。  
- 更精细的动作效率或动作平滑项（若未来可获得历史动作）。  
- 动态课程权重或 stage 门控：若训练后期出现速度-姿态冲突时可考虑 `curriculum_weighting`。

**训练后应关注的 failure modes**  
1. **静止站立或原地踏步**：主奖励与速度强耦合，但 survival_gate 在高稳定性站立时也为 1，agent 可能靠零速避过危险——需观察 episode 长度是否很长但 vx 始终接近 0。  
2. **高度徘徊在门控边缘**：agent 可能会刻意将 body_z 保持在刚好让 gate 足够高又能获取正 vx 的区域，需检查 body_z 分布是否合理。  
3. **快速冲刺后倒地**：gate 衰减虽降低了靠近危险时的奖励，但无法完全杜绝先快后死的行为，尤其 weight 不调整时。  
4. **高频抖动**：虽然加了极小的动作惩罚，但系数极小，agent 仍可能通过高频抖动获得中等速度，需要观察关节扭矩的时间序列是否平滑。  
5. **侧向漂移 / 圆周运动**：v1 未惩罚侧向速度，可能出现侧向运动但依然有正 vx（机器人大角度偏航后向前速度分量仍为正），后期需加入侧向和偏航约束。
