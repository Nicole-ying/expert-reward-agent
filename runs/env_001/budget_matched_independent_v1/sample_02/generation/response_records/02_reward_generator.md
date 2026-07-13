# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward for 2D lander: approach target, decelerate, keep level, settle softly.
    Uses only observable signals; no info/terminal flags.
    """
    # Current state (action already applied)
    x_next = next_obs[0]
    y_next = next_obs[1]
    vx_next = next_obs[2]
    vy_next = next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Previous position for progress
    x_old = obs[0]
    y_old = obs[1]

    # Distance to target (origin)
    dist_old = (x_old ** 2 + y_old ** 2) ** 0.5
    dist_new = (x_next ** 2 + y_next ** 2) ** 0.5

    # 1. Main learning signal: distance reduction (improvement delta)
    distance_progress = dist_old - dist_new

    # 2. Velocity damping: penalize speed, scaled by proximity to target
    speed = abs(vx_next) + abs(vy_next)
    velocity_damping = -0.1 * speed / (1.0 + dist_new)

    # 3. Orientation penalty: keep body horizontal, suppress spinning
    orientation_penalty = -0.1 * (angle_next ** 2) - 0.05 * (angvel_next ** 2)

    # 4. Soft‑landing bonus: joint‑condition proxy when both legs touch,
    #    vehicle is near target, slow, and level
    e = 2.718281828
    dist_factor = e ** (-dist_new ** 2 / 0.5)
    vel_factor = e ** (-(vx_next ** 2 + vy_next ** 2) / 0.1)
    angle_factor = e ** (-angle_next ** 2 / 0.1)
    both_contact = left_contact * right_contact          # binary 0/1

    soft_landing = 0.5 * both_contact * dist_factor * vel_factor * angle_factor

    # Combine
    total = distance_progress + velocity_damping + orientation_penalty + soft_landing

    components = {
        "distance_progress": distance_progress,
        "velocity_damping": velocity_damping,
        "orientation_penalty": orientation_penalty,
        "soft_landing_bonus": soft_landing
    }

    return float(total), components
```

# reward_v1 设计说明

**selected task_family / dynamics_subtype**  
`navigation_goal_reaching` / `goal_approach_and_soft_contact` – 飞行器必须从上方某处出发，在目标垫上稳定着陆。

**selected reward roles**  
根据 `reward_role_decomposition` 选择了四个职责，覆盖主驱动和安全约束，省略了 v1 暂不需要的效率项。

| role | 实现组件 | 信号来源 |
|------|----------|----------|
| `goal_proximity` (mandatory) | `distance_progress` | `obs[0:2]`, `next_obs[0:2]` |
| `velocity_damping` (mandatory) | `velocity_damping` | `next_obs[2], next_obs[3], next_obs[0:2]` |
| `orientation_penalty` (mandatory) | `orientation_penalty` | `next_obs[4], next_obs[5]` |
| `early_settlement_bonus` (conditional) | `soft_landing_bonus` | `next_obs[0:2], next_obs[2:4], next_obs[4], next_obs[6:8]` |

**role‑to‑signal mapping 与公式算子**  

- **goal_proximity** → `improvement_delta`：`dist_old - dist_new`，每步提供正向梯度，推动飞向原点。  
- **velocity_damping** → `dense_state_signal` (随距离衰减的惩罚)：`-w * speed / (1 + dist)`，距离远时对速度惩罚小，鼓励快速接近；靠近时惩罚加大，强制减速，避免过冲。  
- **orientation_penalty** → `quadratic_penalty`：惩罚倾斜角平方和角速度平方，促使姿态水平，防止单腿撞击和翻滚。  
- **early_settlement_bonus** → `joint_condition_proxy`：用连续因子乘积 `both_contact × exp(−dist²/σ) × exp(−v²/σ) × exp(−θ²/σ)` 构造 soft-landing 信号。即使没有显式成功标志，也能在满足稳定着陆条件时提供额外奖励，引导 agent 完成最终停驻。

**excluded roles 及原因**  

- `fuel_efficiency`：v1 默认先学会合理飞行和着陆，燃油效率优化留到后续。  
- `time_penalty`：缺少步骤数/剩余时间信号，且未授权使用 `training_progress`，不可实现。  
- `explicit_success_reward`：`info` 为空，无可用的 success/failure flag，不能依赖。  

**为什么没有 terminal_success_reward / terminal_failure_penalty**  
环境卡片明确 `explicit_success_flag_available=false`，`explicit_failure_flag_available=false`，`info` 固定为空。任何依赖 `info["success"]` 或 `info["termination_reason"]` 的写法都会失败，因此完全放弃稀疏终端奖励，转而用稠密的 `soft_landing_bonus` 近似成功状态。

**留到后续迭代的职责**  

- `fuel_efficiency`：发动机点火惩罚（离散动作代价）。  
- 更精细的健康门控（如 `soft_health_gate` 从姿态/角速度构建，在主奖励恶化时乘性抑制）。  
- 动态课程权重（`curriculum_weighting`）用于 late‑stage 精细控制。  

**训练后应重点观察的 failure modes**  

1. **冲击着陆**：速度惩罚不够强导致高速撞击，应检查 `velocity_damping` 在近距离是否有效压制速度。  
2. **单腿支撑/不稳定**：`soft_landing_bonus` 只奖励双腿接触，若 agent 满足于单腿可能无法获得 bonus，需看姿态和水平速度是否诱导翻滚。  
3. **悬停不前**：`distance_progress` 和速度惩罚冲突，agent 可能宁可静止也不接近目标，需调整 `velocity_damping` 权重或距离门控形状。  
4. **过早坠落到垫外**：agent 未控制水平漂移，需结合 `goal_proximity` 和速度阻尼观察。  
5. **不必要的燃料消耗**：虽然 v1 未显式惩罚点火，但未来加入后会暴露；当前表现可作为基线。
