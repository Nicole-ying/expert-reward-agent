# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x_obs, y_obs = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]

    # Distance to target pad (Euclidean)
    dist_obs = (x_obs**2 + y_obs**2)**0.5
    dist_next = (x_next**2 + y_next**2)**0.5

    # --- Component A: approach progress (improvement_delta, positive only) ---
    # Rewards any reduction in distance, encourages reaching the pad.
    progress = max(0.0, dist_obs - dist_next)
    w_progress = 2.0
    approach_reward = w_progress * progress

    # --- Component B: soft landing speed penalty (quadratic with gate) ---
    # Penalises high velocity only when already close to the target,
    # promoting deceleration for a stable touchdown.
    landing_dist_threshold = 1.0
    gate_landing = max(0.0, 1.0 - dist_obs / landing_dist_threshold)  # active when dist_obs < threshold
    w_speed = 1.0
    speed_sq = vx_next**2 + vy_next**2
    soft_landing_penalty = -w_speed * gate_landing * speed_sq

    # --- Component C: stable upright penalty (quadratic) ---
    # Discourages large body angles that could cause flip or single‑foot contact.
    w_angle = 0.1
    upright_penalty = -w_angle * (angle_next**2)

    # Total reward
    total_reward = approach_reward + soft_landing_penalty + upright_penalty

    components = {
        'approach_reward': approach_reward,
        'soft_landing_penalty': soft_landing_penalty,
        'upright_penalty': upright_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
- **task_family**: navigation_goal_reaching
- **dynamics_subtype**: goal_approach_and_soft_contact  

任务核心是到达目标垫并稳定停靠，通过位置收敛、速度衰减和姿态稳定实现。

## selected reward roles
按照 `reward_role_decomposition` 的优先级，v1 聚焦于三个必选职责：

1. **approach_target** – 主学习信号  
   - 信号：`x_position`, `y_position`  
   - 公式：`improvement_delta` 的变体（只奖不罚的 `max(0, dist_old - dist_new)`）  
   - 驱动 agent 快速向目标垫靠拢。

2. **soft_landing** – 安全/减速约束  
   - 信号：`x_velocity`, `y_velocity`（使用 `next_obs` 的速度）  
   - 公式：二次惩罚 + 与距离关联的 hinge‑gate （`dense_state_signal` hinge 形式）  
   - 仅在 agent 接近垫子时激活，强制减速，避免高速冲击。

3. **stable_upright** – 姿态稳定约束  
   - 信号：`body_angle`（`next_obs`）  
   - 公式：`quadratic_penalty`（`-w * angle^2`）  
   - 全程轻微抑制大幅度倾斜，防止翻倒或单侧着地。

## role_to_signal_mapping 与算子选择

| role            | observations used          | formula operator                         |
|-----------------|----------------------------|------------------------------------------|
| approach_target | `obs[0:2]`, `next_obs[0:2]`| improvement_delta (positive only)        |
| soft_landing    | `dist_obs`, `next_obs[2:4]`| hinge gate × quadratic_penalty           |
| stable_upright  | `next_obs[4]`              | quadratic_penalty                        |

- `improvement_delta` 直接给出进度信号，比负的绝对距离更不易受尺度影响，且正向奖励引导性强。  
- 速度惩罚使用 `gate` 机制：只有当 `dist_obs < 1.0` 时惩罚才生效，避免了过早压制探索，符合 “soft_health_gate” 思路，但这里 gate 控制的是惩罚而非主奖励。  
- 姿态惩罚使用简单的二次项，不给姿态调整造成过度负担。

## excluded roles 及原因
- **fuel_efficiency** – v1 阶段不加推力消耗奖惩；过早引入会使 agent 停滞不前（历史尝试已证明 engine_penalty 得分更差）。  
- **terminal_success_bonus / terminal_failure_penalty** – 环境无显式 success/failure flag，且 info 为空；根据规则 `explicit_success_flag_available=false` 和 `explicit_failure_flag_available=false`，v1 不应加入终端奖励。  
- **contact reward**（双腿接触） – 之前的尝试大量使用接触奖励，但 best_score 仍为负；该类奖励容易诱导 agent 过早单脚或身体触地以刷分，且停靠是状态组合的结果，v1 决定通过位置、速度和姿态奖励自然引导到位。  
- **angular_smoothness / early_contact_penalty** – 属于 avoid_roles，舍弃。

## 与历史上失败结构的差异
先前所有尝试都包含**接触奖励**或**引擎惩罚**，但均未解决 “接近目标后停稳” 的问题（分数始终 < 0）。本方案彻底移除接触奖励和引擎惩罚，回归任务本质的三个连续动力学目标：**靠近、减速、直立**。progress 奖励的形状确保 agent 有明确动力；速度惩罚仅在近处激活，形成自然的“接近‑减速”分工；姿态惩罚维持基本稳定性。这种纯运动学‑姿态驱动的奖励结构是全新的设计假设。

## 哪些职责留到后续迭代
- **燃油经济性**：待 agent 学会可靠停靠后再以小权重加入。  
- **终端成功/失败信号**：若后续版本提供 info 中的判定，可引入 terminal bonus/penalty 强化最终行为。  
- **精确接触约束**：在成功着陆率大幅提升后，可增加对双腿同时接触的软奖励，细化最终接触稳定性。

## 训练后应观察的 failure modes
- **hovering**：agent 害怕接近，在远处徘徊（progress 奖励太低）。此时应检查 `approach_reward` 量级，必要时增大 `w_progress` 或考虑使用潜在塑形。  
- **crash near pad**：尽管有 soft_landing 惩罚，权重或 threshold 不合适导致仍高速撞击。可调整 `landing_dist_threshold` 或 `w_speed`。  
- **oscillating around pad**：progress 只奖励距离减少，可能引导 agent 反复冲过再退回以获取微小的 progress 奖励。若出现，可引入微小的 distance 惩罚（如与绝对距离成比例的负项）或降低 progress 奖励。  
- **inverted flight**：姿态惩罚过小，agent 允许倒飞。可提高 `w_angle`。  
- **spin instability**：角速度未被直接惩罚，但 body_angle 的二次惩罚会间接约束，若仍然出现高频振荡，可在后续迭代加入 angular_velocity penalty。
