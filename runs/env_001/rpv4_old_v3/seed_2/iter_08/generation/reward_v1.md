# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for lunar lander style 2D flyer landing.
    Main goal: reach target platform (x≈0, y≈0) with low velocity, upright.
    Components:
        A: progress (distance reduction towards target)  -- main learning signal
        B: velocity damping (next_obs squared velocity)  -- stability constraint
        C: orientation stabilization (next_obs angle/angular velocity) -- stability constraint
        D: fuel penalty (per action cost)                -- efficiency / mild action cost
    Weights chosen so that A dominates, B/C soft‑constrain, D is light.
    """
    # Extract states
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    
    # Current and next distance to target (Euclidean)
    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    
    # Component A: progress – positive when moving closer to (0,0)
    progress = dist - next_dist  # positive if approaching
    w_progress = 2.0
    component_A = w_progress * progress

    # Component B: velocity damping – penalise large speeds in next step
    # Using squared magnitude to keep gradient smooth
    vx_n, vy_n = next_obs[2], next_obs[3]
    vel_penalty = vx_n**2 + vy_n**2
    w_vel = 0.05
    component_B = -w_vel * vel_penalty

    # Component C: orientation stabilization – penalise tilt and rotation
    angle_n, angvel_n = next_obs[4], next_obs[5]
    ang_penalty = angle_n**2 + angvel_n**2
    w_ang = 0.05
    component_C = -w_ang * ang_penalty

    # Component D: fuel penalty – light cost for any engine use
    # action 0 = no engine, others = engine firing
    fuel_cost = 0.0 if action == 0 else 1.0
    w_fuel = 0.01
    component_D = -w_fuel * fuel_cost

    total_reward = component_A + component_B + component_C + component_D

    components = {
        "progress": component_A,
        "vel_damping": component_B,
        "angle_stab": component_C,
        "fuel_penalty": component_D
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 1. selected task_family / dynamics_subtype
- **task_family**: navigation_goal_reaching  
- **dynamics_subtype**: goal_approach_and_soft_contact  
- **control_type**: discrete  
- **morphology**: 2D 飞行器，主发动机 + 姿态发动机，双腿接触检测

## 2. selected reward roles（按 environment_card.md 的 reward_role_decomposition）
基于环境卡片 mandatory / conditional / avoid 分类，本次 v1 选择了 4 个职责：
- **position_approach**（mandatory）：作为主学习信号，引导 agent 向目标 pad（x≈0, y≈0）靠近。
- **velocity_damping**（mandatory）：轻量约束，抑制高速硬着陆。
- **orientation_stabilization**（mandatory）：保持机身竖直，避免倾斜导致不安全接触。
- **fuel_penalty**（mandatory）：鼓励减少发动机使用，符合次要目标，权重极低以避免抑制必要点火。

**排除的角色及原因**：
- `crash_avoidance`：环境未提供视口边界绝对值，任意阈值会引入噪声或严重误罚，且 speed/angle 约束已间接防止大部分危险状态。
- `soft_contact_encouragement`：需要接近目标且低速的先决条件，v1 先专注基础接近与减速，留待后续加入。
- `success_bonus`、`time_efficiency_penalty`、`progress_speed_bonus`：均因缺少相应信号（无 success flag、无时间进度）被列为 avoid roles，未使用。

## 3. role‑to‑signal mapping 与 formula operators
| role | signals used | formula operator | reasoning |
|------|--------------|------------------|-----------|
| position_approach | obs[0], obs[1], next_obs[0], next_obs[1] | **improvement_delta** (distance reduction) | 每步距离减少即给予正奖励，提供稠密梯度。选用 improvement_delta 而非 dense_state_signal，因为 agent 不能停在远离目标的状态刷分，必须持续接近。 |
| velocity_damping | next_obs[2], next_obs[3] | **quadratic_penalty** (on next speed) | 对下一步速度幅值平方惩罚，平滑且连续，在高速时梯度强。基于 next_obs 使得动作与惩罚直接关联。 |
| orientation_stabilization | next_obs[4], next_obs[5] | **quadratic_penalty** (angle² + ang_vel²) | 与速度惩罚相同模式，保持姿态平稳。选用 quadratic 而非 hinge，因为任务中没有明确的“安全角度上限”，只是希望越小越好。 |
| fuel_penalty | action | **step_constant_penalty** (per‑engine‑firing) | 所有非挂空档动作均附加微小惩罚，反映燃料节省目标。权重极小（0.01），避免 agent 不敢点火。 |

- **为何没有 terminal_success_reward / terminal_failure_penalty**：显式 success/failure flag 在 info 中不可用（explicit_success_flag_available=false），禁止依赖 info 字段推断终止原因。

## 4. 组件权重选择
- w_progress = 2.0：确保主学习信号在数值上贡献最大，推动 agent 向目标移动。
- w_vel = 0.05, w_ang = 0.05：二次惩罚项在高速/大角度时可达数个单位，但与 progress（每步约 0.1–0.5）相比处于次要地位，阻止极端行为而不压制探索。
- w_fuel = 0.01：极轻，仅提供微弱偏向，不会让 agent 因害怕代价而不敢使用发动机。

## 5. 与 previous failed structures 的差异
- 不再使用任何形式的 `success_landing_bonus`（稀疏且触发条件难以可靠判断）。
- 不再使用独立的 `crash_penalty`（边界不明确，易误伤）。
- 主信号改用 **距离缩减**（improvement_delta），而非简单的 `next_obs[1] - obs[1]` 型 progress_shaping，能够同时考虑水平和垂直位移，引导更精准的归航。
- 约束项全部基于 next_obs 的二次惩罚，提供连续梯度，而非二值门控，提升学习稳定性。

## 6. 预期 failure modes 与后续迭代方向
训练后应重点观察：
- **低速漂移、不敢下降**：若 fuel_penalty 或 velocity_damping 过强，agent 可能悬停在高处。可进一步降低 w_vel 或 w_fuel，或后续引入 gentle descent bonus。
- **高速冲撞但仍得正分**：表明 progress 信号未与安全解耦，后期可考虑加入 soft_health_gate，在速度/角度恶化时衰减 progress。
- **忽略姿态、单腿触地**：orientation_stabilization 权重可能需要微调，并配合后续 soft_contact_encouragement 要求双腿同时接触。
- 实现效率优化时可逐步提升 w_fuel 或引入 main_engine 与 orientation_engine 差异化惩罚。

当前的 v1 设计遵循“主学习信号 + 轻量安全约束”的原则，为后续迭代留下清晰的改进空间。