# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x = obs[0]          # horizontal position relative to pad
    y = obs[1]          # vertical position relative to pad
    vx = obs[2]         # horizontal velocity
    vy = obs[3]         # vertical velocity
    angle = obs[4]      # body angle (rad)
    left_contact = obs[6]   # left leg contact flag
    right_contact = obs[7]  # right leg contact flag

    # === Component A: goal-approach reward ===
    # Dense, positive, bounded reward for being close to the pad center.
    distance = (x**2 + y**2)**0.5
    goal_reward = 1.0 / (1.0 + 5.0 * distance)

    # === Component B: soft-landing gate ===
    # Multiplicative gate that dampens goal reward when speed or angle are high.
    speed_sq = vx**2 + vy**2
    speed_gate = 1.0 / (1.0 + 2.0 * speed_sq)
    angle_gate = 1.0 / (1.0 + 1.0 * abs(angle))
    gate = speed_gate * angle_gate

    gated_goal_reward = goal_reward * gate

    # === Component C: leg-contact bonus ===
    # Direct encouragement for both legs touching the pad.
    contact_reward = 0.2 * (left_contact + right_contact)

    total_reward = gated_goal_reward + contact_reward

    components = {
        'gated_goal_reward': gated_goal_reward,
        'contact_reward': contact_reward
    }

    return float(total_reward), components
```

# reward_v1 设计说明

- **selected task_family / dynamics_subtype**  
  `navigation_goal_reaching` / `goal_approach_and_soft_contact`  
  核心任务是让飞行器到达并稳定在中心着陆垫上，属于导航+软着陆。

- **selected reward roles**  
  - `goal_approach`（主学习信号，mandatory）  
  - `soft_landing`（安全/健康约束，mandatory）  
    （`successful_termination_bonus` 因缺少显式成功标志被排除，见下文）

- **role-to-signal mapping**  
  - `goal_approach`：使用 `obs[0], obs[1]` 计算到目标的欧氏距离。  
  - `soft_landing`：  
    - 速度维度：`obs[2], obs[3]`（水平与垂直线速度）构建 soft gate；  
    - 姿态维度：`obs[4]`（机体偏转角）构建 angle gate；  
    - 接触维度：`obs[6], obs[7]`（双腿接触标志）作为独立 additive bonus，引导双足触地。

- **formula operators used**  
  - **goal approach**：`dense_state_signal` 的 bounded 变体 `1/(1 + k * distance)`，提供连续、有界、每步有梯度的正向引导。  
  - **soft landing gate**：`soft_health_gate` 结构，将速度与角度两个因子的乘积作为门控，衰减主奖励，迫使 agent 在追求距离奖励的同时控制姿态与速度。每个 gate 因子采用 bounded 形式 `1/(1 + k * signal^2)`，在恶化时平滑衰减但不大幅惩罚安全范围内的正常波动。  
  - **contact bonus**：`dense_state_signal` 的线性正向形式，激励双腿触地，频率高且直接可观测。

- **excluded roles and reasons**  
  - `successful_termination_bonus`：环境 `explicit_success_flag_available=false`，`info` 为空，无法可靠检测成功终止；若强行用硬阈值在步级颁发高清奖极易被 exploit（反复弹跳）。故 v1 不使用，改为用 `soft_health_gate` 及 `contact_reward` 构造任务完成近似信号。  
  - `fuel_efficiency` (次要目标)：属于效率/动作代价，v1 不加入，留待后续迭代优化。  
  - `terminal_failure_penalty`：同样缺乏显式失败 flag，且失败信号本身已由终止环境提供，不需要在奖励中人工映射。

- **why no terminal_success_reward / terminal_failure_penalty**  
  两种信号都依赖不可用的 info 字段或可靠的终止原因解析。环境仅返回 terminated flag 且没有字符串原因，为避免发明信号，v1 完全依托可观测的观测向量构建稠密、安全的奖励。

- **roles left for future iterations**  
  - 燃料/动作代价：可在 agent 已经学会基本着陆后添加轻度惩罚鼓励节能。  
  - 更精细的姿态接触规划：例如同时要求低角速度、特定阈值内的竖直速度等。  
  - curriculum 权重：若早期探索与后期精细控制冲突明显，可引入 training-progress 调度。

- **post-training failure modes to watch**  
  - **hover / slow drift**：若 goal reward 梯度过弱，agent 可能停滞在远离目标的位置，不再逼近。  
  - **contact spam**：若接触奖励权重过高或速度 gate 太宽容，agent 可能学会快速弹跳刷分。  
  - **crash with full gate**：gate 若对速度/角度惩罚不足，agent 可能高速冲垫后被终止。  
  - **inadequate angle suppression**：gate 参数若不够，agent 可能倾斜着陆导致翻倒。  
  观察这些模式可指导 v2 调整 gate 温度和接触权重。
