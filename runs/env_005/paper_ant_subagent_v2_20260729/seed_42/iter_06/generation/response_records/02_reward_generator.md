# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------
    # Survival bonus: encourages the agent to stay alive
    # ------------------------------------------------------------
    survival_bonus = 0.5

    # ------------------------------------------------------------
    # Body height factor (based on next_obs[0] = body_z)
    # healthy range [0.3, 0.9] with linear ramps to 0.2 and 1.0
    # ------------------------------------------------------------
    z = next_obs[0]
    if z < 0.3:
        height_factor = max(0.0, (z - 0.2) / 0.1)      # 0.2 -> 0, 0.3 -> 1
    elif z <= 0.9:
        height_factor = 1.0
    else:
        height_factor = max(0.0, (1.0 - z) / 0.1)      # 1.0 -> 0, 0.9 -> 1

    # ------------------------------------------------------------
    # Upright factor (based on next_obs[2], next_obs[3] = quat_x, quat_y)
    # body_up_z = 1 - 2*(x^2 + y^2); clip negative to 0
    # ------------------------------------------------------------
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    body_up_z = 1.0 - 2.0 * (quat_x * quat_x + quat_y * quat_y)
    upright_factor = max(0.0, body_up_z)

    # ------------------------------------------------------------
    # Forward velocity (next_obs[13] = body_x_velocity)
    # ------------------------------------------------------------
    forward_vel = next_obs[13]

    # ------------------------------------------------------------
    # Health gate: product of height and upright factors
    # ------------------------------------------------------------
    health_gate = height_factor * upright_factor

    # ------------------------------------------------------------
    # Forward progress: only rewarded when the agent is healthy
    # ------------------------------------------------------------
    forward_progress = forward_vel * health_gate

    # ------------------------------------------------------------
    # Total reward and components
    # ------------------------------------------------------------
    total_reward = survival_bonus + forward_progress
    components = {
        'survival_bonus': survival_bonus,
        'forward_progress': forward_progress,
    }
    return float(total_reward), components
```

# reward_v1 设计说明

- **selected task_family / dynamics_subtype**  
  `locomotion_continuous_control` – 四足机器人通过扭矩控制持续向前移动，同时维持身体高度和直立姿态。

- **selected reward roles**  
  1. **生存激励（survival bonus）**：每步固定正值，鼓励 agent 学会避免摔倒（不触发高度越界终止），是基础安全角色。  
  2. **健康门控前进奖励（forward progress with health gate）**：核心学习信号，驱动机器人在身体姿态安全（高度在舒适区间、躯干接近直立）时最大化前向速度，而在身体状态恶化时逐渐切断前向奖励，引导 agent 恢复健康。

- **role_to_signal_mapping**  
  - survival_bonus：无依赖信号，直接给常量。  
  - forward_progress：依赖 `next_obs[13]`（前向速度）、`next_obs[0]`（身体高度）、`next_obs[2], next_obs[3]`（四元数 x,y 计算直立程度）。

- **每个 role 选择的 formula operator**  
  - `survival_bonus`：固定常数，属于最简单的基本激励。  
  - `forward_progress`：主体使用乘积门控形式，即 `forward_velocity * (height_factor * upright_factor)`。  
    - `height_factor` 采用分段线性 hinge：在安全区间 `[0.3,0.9]` 内恒为 1，超出部分线性衰减到 0（终止边界留 0.1 的余量）。  
    - `upright_factor` 直接用 `max(0, body_up_z)`，抑制倾斜甚至倒立时的前进收益。  
    - 门控乘积实现了 **soft_health_gate** 的语义：不是独立惩罚，而是衰减主任务信号，使 agent 在健康恶化时前进收益平滑下降，避免“先冲后死”。

- **excluded roles 及原因**  
  - **terminal_success_reward / terminal_failure_penalty**：环境中无显式的 success/failure flag，故排除。  
  - **独立高度/姿态/侧向漂移惩罚（hinge/quadratic）**：不再作为独立组件加入，避免与其他惩罚项冲突或强压探索；改为由健康门控统一处理，减少组件数量与调参难度。  
  - **动作代价/效率项**：v1 阶段暂不引入，让 agent 先学会稳定前进，能耗优化留到后续迭代。  
  - **curriculum_weighting**：当前无训练进度参考，且任务无明显的阶段性冲突，不启用。  
  - **joint_condition_proxy 显式替代 forward**：未采用多目标乘积式 proxy，因为简单门控前进已经结合了高度和直立，且存活奖励提供底层安全引导，多因子 arithmetic/geometric proxy 容易塌缩且在这里可能会使训练早期梯度不足。

- **为什么没有使用 terminal_success_reward / terminal_failure_penalty**  
  environment_card 明确标注 `explicit_success_flag_available: false` 及 `explicit_failure_flag_available: false`，因此无法在 v1 中依赖终止信号做奖励。但高度越界可从 `next_obs[0]` 间接推断，利用健康门控在越界前就提供软梯度，远比在终止时才给 spike 更有效。

- **哪些职责留到后续迭代**  
  - 侧向漂移、角速度、关节能耗等精细约束。  
  - 面向更高速度的凸化前向奖励（例如 `forward_vel**2`）或速度饱和函数。  
  - 更复杂的姿态控制（如 pitch/roll 角速度抑制）和对摔倒的特异性早期预警。  
  - 动作平滑性或效率项。

- **训练后应该观察哪些 failure modes**  
  1. **原地站立不动**：生存奖金累积可观，agent 可能满足于 0 前进速度。如果出现，需增大 `forward_progress` 的权重或凸化前向奖励。  
  2. **在健康边界附近震荡**：agent 可能因门控线性衰减而频繁进出衰减区，表现为身体高度或姿态抖动。若发生，可考虑将高度 ramp 宽度加宽或改用更平滑的 sigmoid。  
  3. **侧漂/旋转不受控，但仍前进**：由于未约束侧向速度和偏航角速度，agent 可能歪着走但仍保持直立和高度，奖励依然为正。若此行为导致后续摔倒，可在下一迭代加入轻量侧向漂移 hinge 或偏航角速度 penalty。  
  4. **高速冲撞导致瞬间越界**：如果前向速度奖励过大，agent 可能加速至物理不稳定并摔倒，需要调节权重或在高速时降低门控容忍度。
