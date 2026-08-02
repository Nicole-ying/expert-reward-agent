# 设计理由
本轮修改 **A_progress_gated** 组件。子代理调研与组件表均显示，该组件的进度奖励被 speed_gate 衰减至近乎为零（episode sum 仅 0.55，占比 0.3%），导致下降阶段完全没有有效梯度，agent 仅在着陆点附近依赖 C_landing_steady 徘徊但无法快速抵达。  
**修改内容**：  
1. 将 safe_speed 的自适应阈值由 `0.3 + 0.7*d` 调整为 `0.3 + 1.5*d`，使远距离时允许更高速度（例如 d=2 时 safe_speed=3.3，正常下降速度远低于此值），gate 保持为 1.0。  
2. 将 gate 的衰减函数从 `exp(-2.0*(ratio-1)^2)` 改为更平缓的 `exp(-(ratio-1))`（一次方），确保在轻微超速时仍有梯度，最低 gate 保持在 0.3。  
3. 将潜在差分放大因子由 1.0 提升至 10.0，使每步 progress 峰值约 0.2，与 C_landing_steady 每步 0.20 处于可比数量级。  
整体仍遵循乘积 gate 结构，不引入新的惩罚项，符合"让引导信号重新流动"的校准目标。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- Extract next_obs signals ---
    x = next_obs[0]          # horizontal position relative to pad center
    y = next_obs[1]          # vertical height above pad
    vx = next_obs[2]         # horizontal velocity
    vy = next_obs[3]         # vertical velocity
    angle = next_obs[4]      # body angle (0 upright)
    angular_vel = next_obs[5]  # angular velocity
    left_contact = next_obs[6]  # left leg contact
    right_contact = next_obs[7] # right leg contact

    # --- Helper: distance from pad center (target) ---
    horizontal_dist = abs(x)
    distance_to_target = (horizontal_dist**2 + y**2) ** 0.5

    # --- Component A: main progress signal via potential-based shaping ---
    norm_distance = distance_to_target / 2.5
    angle_penalty = abs(angle) / 1.57
    potential = -(norm_distance + 0.3 * angle_penalty)

    prev_x = obs[0]
    prev_y = obs[1]
    prev_angle = obs[4]
    prev_horizontal_dist = abs(prev_x)
    prev_distance = (prev_horizontal_dist**2 + prev_y**2) ** 0.5
    prev_norm_distance = prev_distance / 2.5
    prev_angle_penalty = abs(prev_angle) / 1.57
    prev_potential = -(prev_norm_distance + 0.3 * prev_angle_penalty)

    potential_delta = potential - prev_potential

    # --- Component B: soft velocity health gate (revised) ---
    speed = (vx**2 + vy**2) ** 0.5
    # Relaxed safe speed: distant positions allow higher velocity
    safe_speed = 0.3 + 1.5 * distance_to_target
    overspeed_ratio = speed / (safe_speed + 1e-6)
    # Smoother attenuation: use linear-exponential with power 1 (gentler decay)
    gate = 0.3 + 0.7 * (2.718281828 ** (-max(0.0, overspeed_ratio - 1.0)))

    # Scale progress to have meaningful per-step magnitude
    scaled_progress = potential_delta * 10.0
    gated_progress = scaled_progress * gate

    # --- Component C: landing steady-state reward (unchanged) ---
    dist_factor = max(0.0, 1.0 - distance_to_target / 0.15)
    contact_factor = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    speed_factor = max(0.0, 1.0 - speed / 0.1)
    angle_factor = max(0.0, 1.0 - abs(angle) / 0.1)
    angular_factor = max(0.0, 1.0 - abs(angular_vel) / 0.5)

    landing_factor = dist_factor * contact_factor * speed_factor * angle_factor * angular_factor
    C_landing = 0.3 * landing_factor

    total_reward = gated_progress + C_landing

    components = {
        'A_progress_gated': gated_progress,
        'C_landing_steady': C_landing
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 所有观测均已使用，无信号缺失；进度组件被过度抑制，稳态奖励主导，属于校准问题。
- **behavior**: agent 在着陆点附近徘徊并持续获得稳态奖励，但未完成真正着陆（多数 episode 超时而非自然终止）。
- **signal**: progress 信号被过于保守的 speed_gate 压至每步 0.0008，下降过程缺乏有效梯度。
- **level**: Level 2
- **hypothesis**: 放宽 speed_gate 允许合理下降速度通过，并放大 progress 尺度，将恢复从起点到目标的有效引导梯度，预期 agent 更快接近垫子，从而在达到稳态区域后完成着陆，提升得分。
- **risk**: 若 gate 在远距离仍不够宽松，progress 可能仍偏弱；若近距离收紧过晚，可能导致撞击着陆垫。若 C_landing_steady 的坚硬接触条件阻碍最终着陆，需后续改造其接触因子为连续形态。