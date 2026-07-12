# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
这是一项连续控制运动任务：控制一个具有 8 个扭矩关节的四足机器人在 3D 环境中向前行进（走路或奔跑），同时必须保持身体高度处于健康范围内（0.2 m ~ 1.0 m）。主要目标是实现稳定的前进速度，而非仅仅保持站立；次要目标可包括运动平滑性、能量效率及减少侧向漂移等。任务不应与简单悬停或固定位置平衡混淆。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control
confidence: high
reason: 核心目标是让四足机器人通过连续扭矩控制产生持续向前的运动，属典型的运动前进控制问题。没有离散到达目标，没有抓取物体，不存在稀疏探索或安全约束下的驾驶进度，且主目标明确（前进），因此归入此任务族。

动力学子类型 dynamics_subtype: multi_legged_body_locomotion
（四足身体沿地面持续前进，腿足与地面交互产生运动）

## 3. 观察空间 observation_space
- type: Box
- shape: (27,)
- dtype: float32 （推断，未明确指定但通常如此）
- 各维度含义与奖励可用性：

| index | 名称               | 含义                                      | reward_usable |
|-------|-------------------|-------------------------------------------|---------------|
| 0     | body_z            | 身体质心高度 (m)                           | true          |
| 1     | quat_w            | 方向四元数实部 w                           | true (用于计算直立度) |
| 2     | quat_x            | 四元数 x 分量                              | true (用于计算直立度) |
| 3     | quat_y            | 四元数 y 分量                              | true (用于计算直立度) |
| 4     | quat_z            | 四元数 z 分量                              | true (用于计算直立度) |
| 5     | joint_1_angle     | 前左髋关节角度                              | true          |
| 6     | joint_2_angle     | 前左踝关节角度                              | true          |
| 7     | joint_3_angle     | 前右髋关节角度                              | true          |
| 8     | joint_4_angle     | 前右踝关节角度                              | true          |
| 9     | joint_5_angle     | 后左髋关节角度                              | true          |
| 10    | joint_6_angle     | 后左踝关节角度                              | true          |
| 11    | joint_7_angle     | 后右髋关节角度                              | true          |
| 12    | joint_8_angle     | 后右踝关节角度                              | true          |
| 13    | body_x_velocity   | 世界坐标系下身体前进速度 (m/s)               | true (主指标) |
| 14    | body_y_velocity   | 世界坐标系下侧向速度 (m/s)                   | true          |
| 15    | body_z_velocity   | 垂直速度 (m/s)                              | true          |
| 16    | body_roll_velocity| 滚转角速度 (rad/s)                          | true          |
| 17    | body_pitch_velocity| 俯仰角速度 (rad/s)                         | true          |
| 18    | body_yaw_velocity | 偏航角速度 (rad/s)                          | true          |
| 19    | joint_1_velocity  | 关节1角速度                                | true          |
| 20    | joint_2_velocity  | 关节2角速度                                | true          |
| 21    | joint_3_velocity  | 关节3角速度                                | true          |
| 22    | joint_4_velocity  | 关节4角速度                                | true          |
| 23    | joint_5_velocity  | 关节5角速度                                | true          |
| 24    | joint_6_velocity  | 关节6角速度                                | true          |
| 25    | joint_7_velocity  | 关节7角速度                                | true          |
| 26    | joint_8_velocity  | 关节8角速度                                | true          |

注：观测中不包含全局 x/y 位置、接触力或任何地面反力信息。

## 4. 动作空间 action_space
- type: Box (continuous)
- shape: (8,)
- 范围：每个关节 [-1.0, 1.0]（扭矩归一化或扭矩值本身）
- 各维度含义：

| action dim | 名称             | 含义                         |
|-----------|------------------|------------------------------|
| 0         | hip_1_torque     | 前左髋关节扭矩               |
| 1         | ankle_1_torque   | 前左踝关节扭矩               |
| 2         | hip_2_torque     | 前右髋关节扭矩               |
| 3         | ankle_2_torque   | 前右踝关节扭矩               |
| 4         | hip_3_torque     | 后左髋关节扭矩               |
| 5         | ankle_3_torque   | 后左踝关节扭矩               |
| 6         | hip_4_torque     | 后右髋关节扭矩               |
| 7         | ankle_4_torque   | 后右踝关节扭矩               |

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无明确成功终止；任务没有设定“到达目标点”等成功条件。
- failure-like termination: 
  1. body_height_outside_healthy_range：身体高度超出 (0.2, 1.0)（过低视为跌倒/倒塌，过高视为异常弹跳或失控）。
  2. state_value_outside_finite_range：任何状态值为 NaN 或无穷。
- ambiguous termination: 无（上述均为失败类终止）
- truncation: 时间限制到达（time_limit_reached），通常为 max_episode_steps 截断，不应视为成功或失败，只能作为“存活但未完成特定目标”。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false （终止标志 `terminated` 可间接视为失败触发，但未在 info 中提供明确 label）
- allowed_info_fields: [] （空，不能从 info 获取任何信息）
- forbidden_or_uncertain_info_fields:
  - reward_forward
  - reward_ctrl
  - reward_contact
  - reward_survive
  - x_position
  - y_position
  - distance_from_origin
  这些字段明确被禁止用于奖励函数设计。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前观测，27维）
- action（当前动作，8维）
- next_obs（下一步观测，27维）
- info 中明确允许的字段（当前为空，即不许使用任何 info 字段）
- training_progress 仅当 prompt 明确允许时才可引入（此处未明确允许，应视为禁用，除非特别说明）

禁止使用：
- original_reward（官方奖励被屏蔽）
- official_reward
- 任何未声明的 info 字段（包括所有 forbidden_info_fields）
- 任何未在观测空间中指定的信号（如全局 x/y 坐标）
- 任何状态累计器（原要求为纯无状态函数）

## 7. 可用于奖励函数的信号
这些信号均可从 obs/next_obs/action 获得：

- position:
  - 身体高度 `body_z` (obs[0])，可评估与理想高度的偏差。
  - 所有关节角度 obs[5:13]，可鼓励或惩罚特定姿态。
- velocity:
  - 前进速度 `body_x_velocity` (obs[13])，直接反映前进表现。
  - 侧向速度 `body_y_velocity` (obs[14])，可用于惩罚侧向漂移。
  - 垂直速度 `body_z_velocity` (obs[15])，可惩罚过度跳跃或不稳定。
  - 身体角速度 (roll/pitch/yaw) obs[16:19]，可惩罚剧烈摇晃。
  - 关节角速度 obs[19:27]，可惩罚关节振动或高能耗。
- orientation:
  - 由四元数 obs[1:5] 可计算 body_up_z = 1 - 2*(quat_x² + quat_y²)（取值范围约 -1 到 1，1 表示完全直立），可用于奖励直立姿态。
- action/engine:
  - 动作扭矩 `action` 的 L2 范数或绝对值和，用于鼓励小力矩（节能）。
  - 动作变化率（需要前后步动作，但 `compute_reward` 只有单步信息，无法直接获前一动作），因此动作平滑（变化惩罚）只能通过两帧动作差来近似，若函数不提供 `prev_action`，该信号不可用。通常环境要求无状态，故可能禁用动作变化惩罚。

- other:
  - 存活信号可用 `body_z` 是否在健康范围内（但范围外的已终止不会调用 reward 函数，所以密集奖励可用到 “接近边界” 的惩罚）。

## 8. 不确定或不可用的信号
- 全局位置 x, y：观测不提供，且被明确禁止。
- 接触力/足端触地状态：未提供。
- 能量/扭矩积分历史：函数无状态无法累积。
- 前一步动作：`compute_reward` 未传入，不能做动作变化惩罚而不违反无状态假设。
- info 中的任何官方奖励项（forward, ctrl, contact, survive）明确禁止。
- 任务进度/里程碑（如“走了多远”）因无 x 坐标信号，无法直接获得基于位置的进度奖励。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: multi_legged_body_locomotion
control_type: continuous
morphology:
  body_type: quadruped (四足)
  actuator_type: torque-controlled (8 joints)
  contact_structure: feet-ground contacts (not observed)
primary_objectives:
  - 维持稳定的向前行进速度（主要通过 body_x_velocity）
  - 保持身体姿态稳定与高度的健康（身体不跌倒、不飞起）
secondary_objectives:
  - 减小侧向偏移（body_y_velocity 接近 0）
  - 降低关节扭矩或功率消耗（动作大小）
  - 维持身体俯仰/滚转平稳（角速度小）
main_failure_risks:
  - 过早跌倒（高度过低）或异常弹跳（高度过高）导致终止
  - 机器人主要使用静立或原地踏步来避免跌倒，但前进速度极低
  - 出现大幅侧向漂移，丧失方向控制
  - 高能耗抖动策略（频繁快速切换扭矩）虽不死却无效率
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_progress
  purpose: 鼓励机器人向前移动，核心任务目标。
  why_required: 若没有前进奖励，智能体可能学会仅保持站立（静止）而避免跌倒，无法完成任务。
  usable_signals: [body_x_velocity]
  risks: 若权重过大且缺乏其他约束，可能产生泡沫弹性前进（剧烈波动丢弃稳定性）。

- role_id: survival_and_health
  purpose: 确保身体高度和方向保持在健康范围内，避免早期跌倒触发终止。
  why_required: 终止本身就是高额惩罚（0 后续奖励），但密集的接近惩罚可更高效引导。
  usable_signals: [body_z, body_up_z (由 quat 计算)]
  risks: 若仅用高度和直立度，可能鼓励僵硬站立而抑制前进探索，需配合前进项。

- role_id: energy_efficiency (optional_but_standard)
  purpose: 惩罚过高的关节扭矩，促进平滑、节能的步态。
  why_required: 纯速度奖励容易产生 bang-bang 控制，加剧能耗却没有额外收益；此职责可修剪不必要的能量损耗。
  usable_signals: [action (L2 norm)]
  risks: 可能压低必要的力导致机器人不移动；需仔细调参，可作为条件职责或与前进奖励联合调整。

### 10.2 条件职责 conditional_roles
- role_id: lateral_stability
  purpose: 抑制侧向速度，迫使机器人沿直线前进。
  condition_to_use: 在训练前期可仅使用主职责，当 agent 已能基本前进且存活一定步数后，再逐渐引入侧向惩罚，避免早期过度约束抑制探索。
  usable_signals: [body_y_velocity]
  risks: 若过早使用，会降低运动多样性；可能偏向极其保守的原地踏步。

- role_id: body_orientation_penalty
  purpose: 进一步惩罚身体俯仰/滚转速度或固定的身体倾斜（已包含在 survival 中，但可强化）。
  condition_to_use: 作为生存职责的细化，当观察到机器人前进但经常偏斜时可打开。
  usable_signals: [body_roll_velocity, body_pitch_velocity, body_up_z]
  risks: 与生存职责功能重叠，可能导致奖励过分集中；需避免过度惩罚自然步态变化。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: action_smoothing (禁止)
  reason: 奖励函数无状态，无法获得前一动作，不能计算动作差分；强行用某种方式估计会导致错误。
  forbidden_or_missing_signals: [prev_action (missing)]

- role_id: contact_force_management (禁止)
  reason: 环境未提供接触力或足端触地信号，无法使用。
  forbidden_or_missing_signals: [contact布尔/力，缺失]

- role_id: distance_progress_reward (禁用)
  reason: 没有全局 x 坐标，也无法从 info 获得 x_position；前进速度本身已间接反映进度，但无法累积距离作为奖励。
  forbidden_or_missing_signals: [x_position, distance_from_origin]

- role_id: official_reward_terms (严格禁止)
  reason: 所有官方 reward 项被屏蔽且被列为禁止使用字段。
  forbidden_or_missing_signals: [info里所有 reward_* 字段]

## 11. role_to_signal_mapping
| role_id | usable signals |



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

