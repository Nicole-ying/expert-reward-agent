# Response Record

# 设计理由
agent 平均 episode 长度 68 步但全部以 terminated 结束，score 低至 -110，内部奖励组件幅度很小，说明外部存在巨大的失败惩罚（如坠毁负分），当前组件无法有效阻止。因此将现有 **velocity_penalty**（全局 speed² 惩罚）改为仅在靠近地面时生效的**垂直下降速度 hinge 惩罚**：对向下的垂直速度超出安全阈值 (0.5) 的部分施加惩罚，并用高度因子门控，确保高空不干预、近地强约束。系数设为 0.3，使得每步典型惩罚（~0.01–0.15）不超过主进展信号（per‑step 约 0.26）的 2 倍，同时给予减速梯度。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- Unpack states ---
    x_pos, y_pos, x_vel, y_vel, body_angle, ang_vel, left_contact, right_contact = obs
    nx_pos, ny_pos, nx_vel, ny_vel, nbody_angle, nang_vel, nleft_contact, nright_contact = next_obs

    # Constants
    current_sq_dist = x_pos ** 2 + y_pos ** 2
    next_sq_dist = nx_pos ** 2 + ny_pos ** 2

    PROXIMITY_SCALE = 10.0
    ANGLE_GATE_K = 15.0
    GROUND_PROXIMITY_SCALE = 4.0
    CONTACT_PROXY_K = 4.0
    FUEL_COST_WEIGHT = 0.15

    # 1. Goal Proximity (mandatory)
    dist_improvement = current_sq_dist - next_sq_dist
    proximity_reward = PROXIMITY_SCALE * dist_improvement / (1.0 + abs(dist_improvement))

    # 2. Orientation Stability Gate
    angle_error = abs(nbody_angle)
    angle_gate = 2.718281828 ** (-ANGLE_GATE_K * angle_error ** 2)
    gated_proximity = proximity_reward * angle_gate

    # 3. Vertical Descent Penalty (hinge, gated by height)
    safe_down_speed = 0.5
    overspeed = max(0.0, -ny_vel - safe_down_speed)       # ny_vel > -0.5 时不惩罚
    ground_proximity = 1.0 / (1.0 + GROUND_PROXIMITY_SCALE * abs(ny_pos))
    vertical_descent_penalty = -0.3 * overspeed * ground_proximity

    # 4. Soft Landing Contact Proxy
    leg_contact_both = min(nleft_contact, nright_contact)
    near_ground_factor = 1.0 / (1.0 + CONTACT_PROXY_K * abs(ny_pos))
    contact_proxy = (leg_contact_both * near_ground_factor) ** 0.5
    soft_landing_bonus = 2.0 * contact_proxy

    # 5. Action Efficiency
    is_engine_on = 0.0 if action == 0 else 1.0
    fuel_cost = -FUEL_COST_WEIGHT * is_engine_on

    total_reward = gated_proximity + vertical_descent_penalty + soft_landing_bonus + fuel_cost

    components = {
        "proximity_reward": gated_proximity,
        "vertical_descent_penalty": vertical_descent_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "fuel_cost": fuel_cost
    }

    return float(total_reward), components
```

# 诊断摘要
- **evidence**: score=-110.44, terminated=20/20, len=68.45；内部组件总和～19，远小于外部负分，说明每次 episode 都遭遇大额失败惩罚。
- **behavior**: agent 可能在接近地面时垂直速度过大，导致坠毁或越界，未实现软着陆。
- **signal**: 缺少针对近地阶段减速的有效梯度；原有 velocity_penalty 幅度太低且无方向性门控。
- **level**: Level 2 —— 将全局速度惩罚改为近地垂直下降速度的 hinge 惩罚（数学形态变换）。
- **hypothesis**: 针对向下速度超出安全阈值的惩罚会迫使 agent 在低空减速，减少坠毁，从而提升外部分数。
- **risk**: 安全阈值设在 0.5 可能过于保守，若实际允许更高速度会导致过早减速，拉长 episode 或降低效率。
