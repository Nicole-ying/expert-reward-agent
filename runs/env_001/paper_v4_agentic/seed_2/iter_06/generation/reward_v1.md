# reward_v1.py

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
    # Euclidean distance to target (x=0, y=0 meaning on the pad)
    distance_to_target = (horizontal_dist**2 + y**2) ** 0.5

    # --- Component A: main progress signal via potential-based shaping ---
    # Potential: combines position error and orientation error.
    # At the goal (upright, centered, low altitude), potential → 0 (highest potential).
    # We define potential as negative distance minus an angle penalty.
    # Normalize distance: max plausible distance ~ sqrt(1.5^2 + 2.0^2) ≈ 2.5, so /2.5 gives ~[0,1]
    norm_distance = distance_to_target / 2.5
    # Angle penalty: absolute angle, normalized by pi (~1.57 rad max before crash)
    angle_penalty = abs(angle) / 1.57
    potential = -(norm_distance + 0.3 * angle_penalty)

    # Previous potential (from obs)
    prev_x = obs[0]
    prev_y = obs[1]
    prev_angle = obs[4]
    prev_horizontal_dist = abs(prev_x)
    prev_distance = (prev_horizontal_dist**2 + prev_y**2) ** 0.5
    prev_norm_distance = prev_distance / 2.5
    prev_angle_penalty = abs(prev_angle) / 1.57
    prev_potential = -(prev_norm_distance + 0.3 * prev_angle_penalty)

    potential_delta = potential - prev_potential
    A_progress = potential_delta  # scale ~ [-0.2, 0.2] per step

    # --- Component B: soft velocity health gate on approach to pad ---
    # Agent must slow down as it gets close. If speed is too high relative to distance,
    # we attenuate the progress reward.
    # Speed norm: sqrt(vx^2 + vy^2). Typical safe landing speed < 0.5.
    speed = (vx**2 + vy**2) ** 0.5
    # Gate threshold: at distance d, safe speed ≈ 0.3 + 0.7*d. If speed >> safe, gate → 0.3.
    safe_speed = 0.3 + 0.7 * distance_to_target
    overspeed_ratio = speed / (safe_speed + 1e-6)
    # Gate factor: 1 when ratio ≤ 1, decays to 0.3 when ratio ≥ 2.5 using smooth exponential
    speed_gate = 0.3 + 0.7 * (2.718281828 ** (-2.0 * max(0, overspeed_ratio - 1.0)**2))

    # Apply gate to progress reward
    gated_progress = A_progress * speed_gate

    # --- Component C: landing steady-state reward ---
    # When the lander is near the pad center, upright, with legs contact, and nearly still,
    # it should receive a continuous positive signal to reinforce staying there.
    # Conditions:
    #   - distance < 0.15 (close to center)
    #   - both legs in contact
    #   - speed < 0.1
    #   - |angle| < 0.1 rad
    # Use a product of soft bounded factors to avoid 0/1 cliff.
    dist_factor = max(0, 1.0 - distance_to_target / 0.15)
    contact_factor = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    speed_factor = max(0, 1.0 - speed / 0.1)
    angle_factor = max(0, 1.0 - abs(angle) / 0.1)
    angular_factor = max(0, 1.0 - abs(angular_vel) / 0.5)

    # Product of factors (only active when all are > 0)
    landing_factor = dist_factor * contact_factor * speed_factor * angle_factor * angular_factor
    C_landing = 0.3 * landing_factor  # modest bonus, does not dominate

    # --- Total reward ---
    total_reward = gated_progress + C_landing

    components = {
        'A_progress_gated': gated_progress,
        'C_landing_steady': C_landing
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
`navigation_goal_reaching` – `goal_approach_and_soft_contact`: 2D 飞行器着陆到目标平台，需要同时满足位置居中、姿态水平、低速和双支撑腿接触。

## selected reward roles
- **goal_proximity (主学习信号)**：通过 potential-based shaping 驱动飞行器持续向目标垫中心靠近，同时兼顾姿态对齐。
- **stability_and_orientation (健康约束)**：用 soft health gate 形式集成到主信号中——当接近速度相对于剩余距离过高时，衰减进度奖励，促使 agent 学会减速逼近。
- **contact_quality / terminal_stability (任务完成近似信号)**：通过联合条件 proxy（软因子乘积），在着陆稳定时提供持续的小额正反馈，强化“到达后保持”行为。

## role_to_signal_mapping
- `goal_proximity`: 使用 `next_obs[0], next_obs[1]` (x, y 位置) 和 `next_obs[4]` (body_angle) 构建归一化 potential。
- `stability_and_orientation`: 使用 `next_obs[2], next_obs[3]` (vx, vy) 和 x, y 计算 speed 与距离，构造门因子。
- `contact_quality / terminal_stability`: 使用 `next_obs[0:8]` 中除位置外的全部维度（速度、角度、角速度、接触信号）联合判断稳定着陆条件。

## formula operator 选择
- **主学习信号**：`potential_based_shaping` (3.3)，将多维度进展（距离减少 + 角度对齐）编码到一个标量 potential 中。采用负的归一化距离与角度惩罚的线性组合。
- **健康门**：`soft_health_gate` (3.5) 的变体，基于 overspeed 比率进行平滑指数衰减，防止 agent 在接近垫面时高速撞击（之前 `velocity_danger` 尝试为独立惩罚，但未解决 crash 问题）。
- **着陆稳态奖励**：`joint_condition_proxy` (3.8) 的乘积形式，将多个连续因子相乘，形成只在所有着陆条件接近满足时才显著非零的软信号。

## excluded roles 及原因
- **energy_efficiency**: v1 阶段首要任务是学会安全的着陆轨迹和稳定停留。燃料效率约束会与探索冲突——agent 需要频繁使用引擎来学习控制；留到 v2 或 v3 优化。
- **terminal_prevention**: 没有独立灾难事件惩罚组件。速度门已经覆盖了导致 crash 的主要前兆（高速接近）；额外的硬惩罚可能压制探索。
- 没有使用 `terminal_success_reward` 或 `terminal_failure_penalty`：因为 `explicit_success_flag_available=false` 且 `info` 不可用。所有成功/失败推断均通过连续观测实现。

## 为何没有使用 terminal_success_reward / terminal_failure_penalty
环境未提供任何显式成功或失败标志 (`info` 为空)，伪造硬终止奖励可能把“步数耗尽截断”误判为成功或失败，造成错误梯度。当前设计将成功信号转化为连续着陆奖励 (`C_landing`)，将失败风险转化为速度门衰减，提供可引导但不会错误覆盖的奖励。

## 留到后续迭代的职责
- **energy_efficiency**：在 agent 能可靠着陆后，增加 `action_efficiency` 算子 (3.7) 以减少多余引擎点燃。
- **复杂的 gate/curriculum**：如果 agent 学会滑行下来但不敢进入接触，可增加对低高度双接触的额外鼓励；若震荡未进入稳态，可对稳态奖励做课程增强。

## 训练后应观察的 failure modes
- **hover oscillation**: agent 在目标上方反复调整悬停，速度足够低但无法下降到接触——需检查 `A_progress_gated` 是否在低距离时仍有足够梯度引导下降。
- **high-speed dive then crash**: 速度门未能充分衰减奖励——可能需要收紧 gate 的 temperature 或降低衰减下限 (0.3→0.1)。
- **one-leg contact bounce**: agent 用单腿接触垫面产生跳跃，利用 `C_landing` 的短暂激活刷分——需降低 `C_landing` 系数或提高接触因子要求（必须双腿）。
- **stand still at spawn**: 如果初始位置恰好靠近中心且宁静，agent 可能选择不动以获取 `C_landing`——需在环境初始化或 episode 早期检查该作弊是否可能，必要时对初始高度/速度设定最小阈值。