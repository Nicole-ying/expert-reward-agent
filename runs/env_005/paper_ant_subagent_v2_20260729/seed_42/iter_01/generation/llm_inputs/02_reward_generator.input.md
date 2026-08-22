# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
机器人需要沿着世界坐标系 x 轴（前进方向）持续、稳定地移动。主要目标是维持稳定的前进行走，而非仅仅保持直立。在实现稳定行走后，更快的向前速度应获得更高评价，但不能以牺牲稳定性为代价。身体高度必须保持在健康范围内，过低（摔倒）或过高（弹跳/飞起）都会导致回合终止。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control
confidence: high
reason: |
  核心目标是"沿前进方向持续稳定移动"，属于典型的前进任务，
  存在明确前进轴（世界 x 轴）。虽然保持直立是生存条件，但
  任务描述明确强调"应维持稳定前进而非仅仅保持直立"，说明
  速度/位移是主目标，直立只是健康约束。

## 3. 观察空间 observation_space
- type: Box
- shape: [27]
- dtype: 连续浮点（推测 float32/64）
- obs[0]: body_z，身体垂直位置（高度），reward_usable: true
- obs[1]: quat_w，身体朝向四元数实部，reward_usable: true（需转换为 uprightness）
- obs[2]: quat_x，身体朝向四元数第一虚部，reward_usable: true（需转换为 uprightness）
- obs[3]: quat_y，身体朝向四元数第二虚部，reward_usable: true（需转换为 uprightness）
- obs[4]: quat_z，身体朝向四元数第三虚部，reward_usable: true（需转换为 uprightness）
- obs[5]: joint_1_angle，第一髋关节角度，reward_usable: true（可用于动作平滑/能耗）
- obs[6]: joint_2_angle，第一踝关节角度，reward_usable: true（可用于动作平滑/能耗）
- obs[7]: joint_3_angle，第二髋关节角度，reward_usable: true（可用于动作平滑/能耗）
- obs[8]: joint_4_angle，第二踝关节角度，reward_usable: true（可用于动作平滑/能耗）
- obs[9]: joint_5_angle，第三髋关节角度，reward_usable: true（可用于动作平滑/能耗）
- obs[10]: joint_6_angle，第三踝关节角度，reward_usable: true（可用于动作平滑/能耗）
- obs[11]: joint_7_angle，第四髋关节角度，reward_usable: true（可用于动作平滑/能耗）
- obs[12]: joint_8_angle，第四踝关节角度，reward_usable: true（可用于动作平滑/能耗）
- obs[13]: body_x_velocity，身体世界 x 轴速度（前进速度），reward_usable: true（主信号核心）
- obs[14]: body_y_velocity，身体世界 y 轴速度（侧向速度），reward_usable: true（抑制侧向漂移）
- obs[15]: body_z_velocity，身体垂直速度，reward_usable: true（监测坠落/弹跳）
- obs[16]: body_roll_velocity，翻滚角速度，reward_usable: true（稳定性指标）
- obs[17]: body_pitch_velocity，俯仰角速度，reward_usable: true（稳定性指标）
- obs[18]: body_yaw_velocity，偏航角速度，reward_usable: true（前进方向稳定性）
- obs[19]: joint_1_velocity，第一髋关节角速度，reward_usable: true
- obs[20]: joint_2_velocity，第一踝关节角速度，reward_usable: true
- obs[21]: joint_3_velocity，第二髋关节角速度，reward_usable: true
- obs[22]: joint_4_velocity，第二踝关节角速度，reward_usable: true
- obs[23]: joint_5_velocity，第三髋关节角速度，reward_usable: true
- obs[24]: joint_6_velocity，第三踝关节角速度，reward_usable: true
- obs[25]: joint_7_velocity，第四髋关节角速度，reward_usable: true
- obs[26]: joint_8_velocity，第四踝关节角速度，reward_usable: true

## 4. 动作空间 action_space
- type: Box
- shape: [8]
- n: 8 维连续动作
- action[0]: hip_1_torque，第一个髋关节扭矩，范围 [-1.0, 1.0]
- action[1]: ankle_1_torque，第一个踝关节扭矩，范围 [-1.0, 1.0]
- action[2]: hip_2_torque，第二个髋关节扭矩，范围 [-1.0, 1.0]
- action[3]: ankle_2_torque，第二个踝关节扭矩，范围 [-1.0, 1.0]
- action[4]: hip_3_torque，第三个髋关节扭矩，范围 [-1.0, 1.0]
- action[5]: ankle_3_torque，第三个踝关节扭矩，范围 [-1.0, 1.0]
- action[6]: hip_4_torque，第四个髋关节扭矩，范围 [-1.0, 1.0]
- action[7]: ankle_4_torque，第四个踝关节扭矩，范围 [-1.0, 1.0]

> 动作均为关节扭矩（torque），值为归一化到 [-1, 1] 后的控制信号。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无明显明确成功终止。环境可能一直运行直到 truncation 或失败终止。
- failure-like termination:
  - body_height_outside_healthy_range: 身体高度超出 [0.2, 1.0]，过低（摔倒/坍塌）或过高（弹跳/飞起）。
  - state_value_outside_finite_range: 任意状态值变为 NaN 或无穷大（仿真崩溃/数值溢出）。
- ambiguous termination:
  - 无显式成功信号。如果 episode 在终止前未触发 failure，可能是在稳定行走中因步数上限 truncated。
- truncation: time_limit_reached（步数上限截断）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false (failure 仅通过 terminated 标志间接体现，但不能区分失败类型)
- allowed_info_fields: []（无可用 info 字段）
- forbidden_or_uncertain_info_fields:
  - reward_forward, reward_ctrl, reward_contact, reward_survive（官方奖励分量，禁止使用）
  - x_position, y_position, distance_from_origin（绝对位置信息，禁止使用）

> 虽然没有显式 flag 可用，但可通过观测信号间接推断失败：
> - 摔倒/坍塌: body_z（obs[0]）急剧下降并接近或低于 0.2，或前一帧 body_z 明显偏高但被截断前突然更低。
> - 弹跳/飞起: body_z 突然超过 1.0 或 body_z_velocity 为正异常值。
> - 方向失控/倒地: body_up_z（由四元数计算）偏离 1.0 很严重，且 body_roll/pitch_velocity 急剧变化。
> - 仿真崩溃: 观测值出现 NaN/Inf 在终止前可能无法读取，但可提前检测。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs: 当前时刻观测向量（27D），在 step 之前
- action: 执行的动作向量（8D）
- next_obs: 下一时刻观测向量（27D），在 step 之后
- info: 空字典 {}，无任何可用字段
- training_progress: 表示训练进度的标量（仅当 prompt 明确允许此 env 使用时才用）

禁止使用：
- original_reward: 屏蔽的官方奖励
- official_reward 及任何从 info 泄露的官方奖励分量（reward_forward 等）
- info 中的任何字段（当前 info 为空）
- obs 或 next_obs 中未声明的维度
- 绝对位置信息（x_position, y_position, distance_from_origin），不在观测空间中，不能用

## 7. 可用于奖励函数的信号
- position:
  - body_z (obs[0]): 身体高度，可判断是否在健康范围 [0.2, 1.0]
  - joint_angles (obs[5:13]): 关节角度，可用于动作平滑性或姿态评估
- velocity:
  - body_x_velocity (obs[13]): 前进速度（主信号核心）
  - body_y_velocity (obs[14]): 侧向速度（应抑制漂移）
  - body_z_velocity (obs[15]): 垂直速度（可检测弹跳/坠落趋势）
  - body_roll_velocity (obs[16]): 翻滚角速度（稳定性）
  - body_pitch_velocity (obs[17]): 俯仰角速度（稳定性）
  - body_yaw_velocity (obs[18]): 偏航角速度（方向稳定性）
  - joint_velocities (obs[19:27]): 关节角速度
- orientation:
  - body orientation quaternion (obs[1:5]): 可计算 body_up_z = 1 - 2*(quat_x² + quat_y²)，范围[-1, 1]，1 表示完全直立。可用于评估是否倾斜过度。
- contact: 无接触力或接触标志信号可用
- action/engine:
  - action (8D): 关节扭矩控制信号，可用于动作效率（能耗）奖励
- other:
  - derived_possible: 失败推断信号（见 5.2），通过 body_z、四元数、速度异常变化间接推测终止原因。

## 8. 不确定或不可用的信号
- 绝对世界 x 位置或里程计: 不可用（被刻意排除）
- 接触力/足端接触标志: 不可用
- 重心投影或 ZMP: 不可用
- 能耗或功率测量: 不可用
- 明确的失败标志（如 info["fell"]）: 不可用
- 明确的成功标志: 不存在
- 目标位置/朝向: 不存在

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: multi_legged_body_locomotion
control_type: continuous
morphology:
  body_type: 3D quadruped with 4 legs
  actuator_type: 8 torque-controlled joints (hip + ankle per leg)
  contact_structure: intermittent ground contacts with unknown contact dynamics
primary_objectives:
  - sustained forward locomotion along world x-axis
  - maintain body stability (avoid falling or launching)
secondary_objectives:
  - increase forward speed (velocity magnitude along x)
  - minimize lateral drift (body_y_velocity)
  - minimize excessive actuation (energy or torque efficiency, if task explicitly requires)
  - maintain smooth joint motions
main_failure_risks:
  - body collapse: body_z falls below 0.2 (robot falls down)
  - body launch: body_z exceeds 1.0 (robot jumps too high)
  - orientation instability: excessive roll/pitch causing loss of control
  - numerical divergence: state values become NaN or Inf
  - inefficient gait: agent fails to develop coordinated forward movement, spins or stays in place
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_velocity
  purpose: 驱动机器人沿世界 x 轴前进
  why_required: 核心任务目标，没有向前速度就没有完成任务
  usable_signals: [body_x_velocity (obs[13])]
  risks: 可能导致 agent 以不稳定方式最大化速度（如弹跳前进），需配合健康门控

- role_id: health_gate
  purpose: 确保机器人保持健康姿态（身体高度安全、基本直立）时才给予前进奖励
  why_required: 防止 agent 为了速度牺牲稳定性（摔倒或弹跳）。若无此 gate，agent 可能在尝试摔倒的过程中获得前进速度的奖励
  usable_signals:
    - body_z (obs[0]) -- 检查是否在 [0.25, 0.9] 等安全子区间
    - body_up_z (derived from obs[1:5]) -- 检查是否接近 1.0（直立）
  risks: 门控太严格会阻止 agent 学习正常行走中的微小波动；需用连续型 bounded_signal 而非 binary gate，允许一定容忍度

### 10.2 条件职责 conditional_roles
- role_id: uprightness_penalty
  condition_to_use: 当任务描述明确要求"稳定行走"且 body_up_z 明显偏离 1.0 时
  usable_signals: [body_up_z (derived from quaternion)]
  risks: 若惩罚过强可能抑制正常的身体摆动（行走中必然有轻微倾斜）；推荐 hinge 形式：只在超出安全区间时惩罚

- role_id: lateral_drift_penalty
  condition_to_use: 当频繁出现侧向漂移（y 轴速度较大）影响前进效率时
  usable_signals: [body_y_velocity (obs[14])]
  risks: 轻微侧向摆动是正常步态，过度惩罚会妨碍正常行走模式

- role_id: action_smoothness
  condition_to_use: 当训练出的步态存在剧烈抖动、关节扭矩跳变严重影响仿真稳定性或能耗要求明确时
  usable_signals: [action (当前), prev_action (需要外部保存)]
  risks: 过度平滑会抑制敏捷动作；仅在 task description 提到效率或步态质量要求时使用

- role_id: vertical_stability_penalty
  condition_to_use: 当 body_z_velocity 或 body_z 波动剧烈，出现弹跳趋势时
  usable_signals: [body_z_velocity (obs[15])]
  risks: 可与 health_gate 合并，但可单独作为阻尼项

- role_id: roll_pitch_stability
  condition_to_use: 当身体翻滚/俯仰角速度过大导致失衡风险时
  usable_signals: [body_roll_velocity (obs[16]), body_pitch_velocity (obs[17])]
  risks: 仅在不稳定时惩罚，使用 hinge 形式，避免影响正常步态

### 10.3 慎用/禁用职责 avoid_roles
- role_id: action_cost (energy efficiency)
  reason: 任务描述未明确要求节能或最小化扭矩。四足行走本身需要较大扭矩，盲目惩罚可能妨碍步态合成。仅当环境额外信息或后续训练显示动作过大导致仿真不稳定时才考虑弱惩罚
  forbidden_or_missing_signals: [无明确能量消耗信号，仅可从 action 幅度间接推断]

- role_id: terminal_success_bonus
  reason: 无显式"到达终点"概念，success termination 不存在。若强行用 forward velocity 累计推断里程并发放 bonus，可能鼓励 agent 延长 episode 但不保证稳定
  forbidden_or_missing_signals: [无 dist_to_goal, 无 x_position]

- role_id: terminal_failure_penalty
  reason: 无显式 failure flag，仅能从 terminated 和 body_z 推断。但 state 在终止步可能已经失效（NaN/Inf），不宜直接读 next_obs 发放惩罚。可考虑用 body_z 趋近 0.2 时给出软 penalty
  forbidden_or_missing_signals: [explicit failure flag]

- role_id: velocity_tracking_to_target
  reason: 无目标速度规范，任务未要求跟踪特定速度
  forbidden_or_missing_signals: [target_velocity]

## 11



# expert_reward_context.md

# Expert Schema Context（非检索版）

这份内容不是 RAG 检索结果，也不是按 benchmark 名称写死的奖励模板。它是给 Reward Generator 使用的固定专家 Schema：先读 environment_card.md 中的任务画像和奖励职责拆解，再从下面的小型公式算子库中选择合适数学形式。

核心顺序必须是：

```text
环境事实 → 任务画像 → 奖励职责 reward roles → 职责-信号映射 → 公式算子 → reward code
```

不要反过来先套某个 skeleton 名称。模板只提供专家思考方式，不构成封闭候选集合。

---

## 1. Expert Schema 使用规则

- environment_card.md 中的 `expert_task_profile`、`reward_role_decomposition`、`role_to_signal_mapping` 优先级最高。
- 本文件只提供通用公式算子，不替代环境卡片。
- 先选 role，再选 signal，再选 formula operator，最后写 compute_reward。
- 如果某个 role 需要的信号不可用，必须排除，不得硬写。
- 如果任务画像与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- 不要因为模板中出现某个 role，就机械加入该 role。
- reward_v1 优先覆盖主学习信号和必要健康约束；效率、能耗、复杂门控和动态权重默认留到后续迭代。

---

## 2. Formula Operator Library

每个算子包含：数学形式、适用场景、触发证据、反模式。

### 2.1 dense_state_signal
- 适用职责：持续前进、速度、姿态、高度、接近目标等连续状态职责。
- 常见形式：
  - positive (线性): `w * signal`
  - positive (凸化): `w * signal**2` 或 `w * exp_form`
    凸化形式在 signal 较大时提供更强梯度。触发证据：episode 长度正常但 score 停滞在低水平，且该信号的 episode_sum_mean 始终偏小——说明 agent 满足于低水平稳态，需要凸化奖励来打破。
  - penalty (二次): `-w * error**2`
  - penalty (hinge): `-w * max(0, threshold - signal)` 或 `-w * max(0, signal - upper)`
    hinge 只在超出安全区间时生效，避免在安全范围内持续惩罚正常波动。触发证据：约束组件的 active_rate≈100% 但 terminated 率仍然很高——说明"全时惩罚"没有给 agent 安全探索空间，它无论怎么调整都被罚。
- 使用条件：该状态信号每步可观测，且与任务目标直接相关。
- 风险：线性正奖励可能导致慢速平台；凸化形式若权重过大可能诱导极端行为；hinge 的 threshold 设太宽则防护不足。

### 2.2 bounded_signal
- 适用职责：限制速度、距离、姿态误差或其他连续信号的极端值。
- 常见形式：
  - 平滑压缩: `x / (1 + abs(x))`
  - 倒数衰减: `1 / (1 + k * abs(error))`
  - 线性衰减: `max(0, 1 - abs(error) / threshold)`
- 使用条件：原始信号可能过大、尺度不稳定，或信号容易被刷分。
- 触发证据：某个信号的 episode_sum_mean 出现极端值（远大于其他组件），说明无界形式被 exploit。
- 风险：threshold 过小会导致反馈饱和或无梯度。
- 反模式：不要用 bounded_signal 替代 hinge penalty——如果目标是"只在越界时惩罚"，用 dense_state_signal 的 hinge 形式，不要用 bounded 包围。

### 2.3 improvement_delta
- 适用职责：接近目标、距离减少、状态改善。
- 常见形式：
  - `old_measure - new_measure`
  - `next_value - current_value`
- 使用条件：obs 和 next_obs 中存在可比较的当前量与下一步量。
- 触发证据：有明确的目标度量（如到目标的距离）且该度量在 episode 中单调递减时 agent 表现好。
- 风险：目标附近可能震荡；没有明确目标度量时不要使用。
- 反模式：不要对速度类信号用 improvement_delta——持续速度本身已经是"进步"，delta 会退化为噪声。

### 2.4 potential_based_shaping
- 适用职责：有明确 potential function 的任务塑形。
- 常见形式：`gamma * Phi(next_obs) - Phi(obs)`
- 使用条件：能够从环境信号定义合理的 Phi。
- 风险：错误 Phi 会误导策略；reward_v1 不默认使用，除非任务天然适合。

### 2.5 quadratic_penalty
- 适用职责：姿态误差、角速度、动作幅度、速度等轻量约束。
- 常见形式：`-w * error**2` 或 `-w * sum(action_i**2)`
- 使用条件：约束信号可观测，且不应压制主学习信号。
- 风险：权重过大会导致 agent_afraid_to_move 或 over_conservative_policy。
- 触发证据：某维度出现高频大幅波动或极端值，但没有触发终止——说明需要轻量抑制而非硬约束。
- 反模式：不要对"有明确安全边界"的信号用 quadratic_penalty（如身体高度必须在 0.2-1.0）。quadratic 从中心开始罚，会让 agent 困在中心不敢动；应改用 hinge 形式只在边界附近生效。

### 2.6 soft_health_gate
- 适用职责：让主进展奖励在健康状态下充分生效，而不是直接加大惩罚。
- 常见形式：`main_reward * gate_factor`，gate_factor 在身体状态恶化时从 1 平滑衰减到 0。
  - 倒数门: `1 / (1 + k * abs(posture_error))`
  - 线性衰减门: `max(0, min(1, (signal - danger) / margin))`
- 使用条件：terminated 主要由健康/安全违规导致，且主奖励在失败回合中仍然显著为正。
- 触发证据（关键）：terminated 率高（>50%）且主进展信号在失败回合的 episode_sum 仍然 >0——说明 agent 在"先冲后死"，需要 gate 在健康恶化时切断主奖励，而不是加一个独立惩罚。
- 风险：gate 太严格会抑制探索；gate 的衰减区间应设在"接近危险但尚未终止"的范围内。
- 反模式：不要用"加大独立惩罚系数"替代 gate。如果 terminated 是因为身体状态越界，单纯加大该状态的惩罚（Level 1）通常不如将其作为 gate 乘到主奖励上（Level 2），因为惩罚只在越界后才生效，gate 在越界前就开始衰减主信号。

### 2.7 joint_condition_proxy
- 适用职责：多个条件必须同时满足的软完成近似，例如 near + low speed + stable。
- 常见形式：`factor_1 * factor_2 * factor_3`，每个 factor 都是连续 bounded 形式。
- 使用条件：没有显式 success flag，但有连续信号可构造 soft proxy。
- 触发证据：agent 能在各个子条件上分别取得进展，但无法同时满足——说明缺一个"联合满足"的引导信号。
- 风险：乘积容易塌缩（一个 factor 趋近 0 则整体为 0）；使用 `(factor_1 + factor_2 + ...) / n` 或几何平均 `(factor_1 * factor_2 * ...) ** (1/n)` 可缓解。
- 反模式：不要用二值条件做乘积——每个 factor 必须是连续函数，否则乘积退化为稀疏信号。

### 2.8 curriculum_weighting
- 适用职责：早期探索和后期精细控制明显冲突时。
- 常见形式：`early_weight = 1 - training_progress`，`late_weight = training_progress`
- 使用条件：training_progress 明确允许，且确有阶段性需求。
- 风险：增加消融混杂；reward_v1 默认不要使用。

---

## 3. 迭代修改时的算子切换指南

以下映射帮助 reflection agent 从"训练反馈证据"直接定位到"该选哪个算子做 Level 2 变换"。
不要求组件名完全匹配；以数学语义和训练表现证据为准。

| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |

