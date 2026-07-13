# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解包观测变量
    x_pos        = obs[0]
    y_pos        = obs[1]
    x_vel        = obs[2]
    y_vel        = obs[3]
    body_angle   = obs[4]
    ang_vel      = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    # 1. 主学习信号：负的欧氏距离，驱动智能体靠近目标垫中心
    distance = (x_pos**2 + y_pos**2)**0.5
    proximity_reward = -1.0 * distance

    # 2. 软着陆速度惩罚：使用距离门控，在接近目标时增强减速要求
    gate = 1.0 / (1.0 + distance)          # 距离越大门越小，避免阻碍早期探索
    speed_sq = x_vel**2 + y_vel**2
    velocity_penalty = -2.0 * speed_sq * gate

    # 3. 稳定性约束：惩罚过大的机体倾斜和旋转速度
    stability_penalty = -0.5 * abs(body_angle) - 0.1 * (ang_vel**2)

    # 4. 双足接触激励：只有双脚都接触垫子时才给予正向奖励
    contact_reward = 2.0 * (left_contact * right_contact)

    # 总奖励
    total_reward = proximity_reward + velocity_penalty + stability_penalty + contact_reward

    # 组件字典，仅包含被累加到 total_reward 中的项
    components = {
        'proximity_reward': proximity_reward,
        'velocity_penalty': velocity_penalty,
        'stability_penalty': stability_penalty,
        'contact_reward': contact_reward
    }

    return total_reward, components
```

# reward_v1 设计说明

- **selected task_family / dynamics_subtype**: `navigation_goal_reaching` / `goal_approach_and_soft_contact`  
  2D 飞行器需要通过姿态与主引擎控制，靠近中央目标垫，完成低速、稳定、双脚接触的停靠。

- **selected reward roles**（基于 `reward_role_decomposition` 的 mandatory roles，按 v1 阶段裁剪）：
  - `proximity_to_target`：负欧氏距离，作为唯一的**主学习信号**（formula operator: `dense_state_signal` 线性形式）。
  - `soft_landing_velocity`：距离门控的速度平方惩罚（formula operator: `dense_state_signal` hinge/penalty 变体 + 门控因子，类似 `soft_health_gate` 思路），仅在目标附近高强度生效。
  - `stability`：姿态角绝对值 + 角速度平方的合并约束（formula operator: `quadratic_penalty` 类）。
  - `dual_contact_incentive`：双脚接触乘积的二值正向奖励（formula operator: 简化 `joint_condition_proxy` 但保持连续——乘积为 0 或 1，符合 v1 简单性）。
  - **budget**: 1 个主信号 + 2 个安全/稳定约束 + 1 个接触引导，共 4 个组件，符合 v1 推荐（2~4）。

- **role_to_signal_mapping**：
  - `proximity_to_target` ← `x_position`, `y_position`
  - `soft_landing_velocity` ← `x_velocity`, `y_velocity`，并利用 `x_position`, `y_position` 构建距离门
  - `stability` ← `body_angle`, `angular_velocity`
  - `dual_contact_incentive` ← `left_support_contact`, `right_support_contact`

- **selected formula operators**：
  - 距离：`-w * distance`（线性，无压缩，因初始距离有限，梯度稳定）
  - 速度：`-w * speed² * gate`，其中 `gate = 1/(1+distance)`（hinge/门控惩罚，远距离时惩罚小，近距离时惩罚全效）
  - 稳定性：`-w1 * |body_angle| - w2 * angular_velocity²`（混合二次和绝对值，轻量约束）
  - 接触：`+w * (left_contact * right_contact)`（二值但直接，利用连续乘积实现）

- **excluded roles 及原因**：
  - `fuel_efficiency`：v1 阶段避免引入动作惩罚，降低探索阻力；后续迭代再加入。
  - `landing_zone_soft_constraint`、`settlement_bonus`：属于 conditional roles，需要精细门限设计，留到 v1 有效后作为增强。
  - `time_penalty`、`survival_bonus`：环境卡片明确列为 avoid_roles，无可用信号且与任务目标冲突。
  - `terminal_success_reward`、`terminal_failure_penalty`：环境没有显式成功/失败标志，且 `info` 为空，禁止使用。

- **为什么没有使用 terminal_success_reward / terminal_failure_penalty**：  
  环境卡片指出 `explicit_success_flag_available=false`，`info` 无任何字段，智能体必须从观测中推断任务完成状态。因此 v1 通过距离、速度、姿态、接触四类稠密信号间接引导成功行为，不依赖离散终止信号。

- **哪些职责留到后续迭代**：
  - 燃料效率（`fuel_efficiency`）
  - 更精确的着陆区强化约束（`landing_zone_soft_constraint`）
  - 稳定停靠的一次性或持续高额奖励（`settlement_bonus`）
  - 若出现高速冲撞但仍拿到正奖励，可考虑引入 `soft_health_gate` 切断主信号

- **训练后应该观察的 failure modes**：
  - agent 可能学习到快速冲向目标垫但来不及减速，导致 crash——观察速度惩罚在终止前是否足够强，必要时调大 `velocity_penalty` 权重或改用更锐利的门控（如 `1/(1+distance²)`）。
  - agent 可能只用单脚接触或悬空假稳定——观察 `contact_reward` 占比和是否出现 “hover” 现象；若单一接触稳定未被惩罚，迭代时可引入缺少双腿接触的小惩罚。
  - 姿态震荡：如果角速度频繁切换，`stability_penalty` 需微调权重，或改用平方项统一惩罚。
  - 整体表现停滞在距离较大处：可能是速度惩罚在远处仍过强，可考虑将 `gate` 的常数分母增大（如 `1 + 5*distance`），或降低 `velocity_penalty` 权重。