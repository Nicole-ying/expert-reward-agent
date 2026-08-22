# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
本环境要求一个 3D 四足机器人通过 8 个连续扭矩关节控制前进，核心目标是**持续向前行走或奔跑**。在前进过程中，机器人必须保持身体高度在健康范围内（0.2 ~ 1.0 m，用 body_z 表示），并尽量保持直立姿态。若高度超出范围或任意状态量变为无效值，回合立即终止。机器人需要**同时兼顾前进速度、身体高度维持和姿态稳定性**，而不是仅保持站立不动。由于无法获取全局 x/y 位置，只能基于瞬时前向速度进行闭环控制，不得依赖累积距离。

## 2. 任务类型选择
selected_route_id: `locomotion_continuous_control`  
confidence: high  
reason: 核心目标是持续前进通过地形（walk or run forward），高度约束是健康的必要条件而非唯一目标，且不存在导航目标点、抓取操作或稀疏探索事件，完全属于典型的多足连续运动控制。

## 3. 观察空间 observation_space
- type: Box  
- shape: (27,)  
- dtype: float32  
- 每个维度含义：

| 索引 | 名称 | 含义 | reward_usable |
|------|------|------|--------------|
| 0 | body_z | 身体中心在世界 z 轴的高度 | true |
| 1 | quat_w | 身体姿态四元数实部 w | true |
| 2 | quat_x | 四元数虚部 x | true |
| 3 | quat_y | 四元数虚部 y | true |
| 4 | quat_z | 四元数虚部 z | true |
| 5 | joint_1_angle | 第 1 髋关节角度 | true |
| 6 | joint_2_angle | 第 1 踝关节角度 | true |
| 7 | joint_3_angle | 第 2 髋关节角度 | true |
| 8 | joint_4_angle | 第 2 踝关节角度 | true |
| 9 | joint_5_angle | 第 3 髋关节角度 | true |
| 10 | joint_6_angle | 第 3 踝关节角度 | true |
| 11 | joint_7_angle | 第 4 髋关节角度 | true |
| 12 | joint_8_angle | 第 4 踝关节角度 | true |
| 13 | body_x_velocity | 世界系 x 方向线速度（前向） | true |
| 14 | body_y_velocity | 世界系 y 方向线速度（横向） | true |
| 15 | body_z_velocity | 世界系 z 方向线速度（竖直） | true |
| 16 | body_roll_velocity | 绕 x 轴角速度 | true |
| 17 | body_pitch_velocity | 绕 y 轴角速度 | true |
| 18 | body_yaw_velocity | 绕 z 轴角速度 | true |
| 19 | joint_1_velocity | 第 1 髋关节角速度 | true |
| 20 | joint_2_velocity | 第 1 踝关节角速度 | true |
| 21 | joint_3_velocity | 第 2 髋关节角速度 | true |
| 22 | joint_4_velocity | 第 2 踝关节角速度 | true |
| 23 | joint_5_velocity | 第 3 髋关节角速度 | true |
| 24 | joint_6_velocity | 第 3 踝关节角速度 | true |
| 25 | joint_7_velocity | 第 4 髋关节角速度 | true |
| 26 | joint_8_velocity | 第 4 踝关节角速度 | true |

## 动作空间 action_space
- type: Box  
- shape: [8]  
- continuous 动作，每个维度范围 [-1.0, 1.0]

| 动作维度 | 名称 | 含义 |
|----------|------|------|
| 0 | hip_1_torque | 第 1 髋关节扭矩 |
| 1 | ankle_1_torque | 第 1 踝关节扭矩 |
| 2 | hip_2_torque | 第 2 髋关节扭矩 |
| 3 | ankle_2_torque | 第 2 踝关节扭矩 |
| 4 | hip_3_torque | 第 3 髋关节扭矩 |
| 5 | ankle_3_torque | 第 3 踝关节扭矩 |
| 6 | hip_4_torque | 第 4 髋关节扭矩 |
| 7 | ankle_4_torque | 第 4 踝关节扭矩 |

## 5. step 与终止条件分析

### 5.1 终止模式
- success-like termination: 无显式成功终止；environment 仅通过时间截断 `truncated` 结束，此时若机器人在高度边界内且不崩溃，可视为存活成功，但不构成硬 success flag。
- failure-like termination:  
  - `body_height_outside_healthy_range`: 身体高度 body_z ≤ 0.2 或 ≥ 1.0，表示摔倒/弹飞。  
  - `state_value_outside_finite_range`: 任一状态量变为 NaN 或 Inf，这是数值崩溃。  
- ambiguous termination: 无。  
- truncation: 达到时间上限，是唯一正常结束方式。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: []  (info 为空字典，无任何字段)  
- forbidden_or_uncertain_info_fields:  
  - `reward_forward`, `reward_ctrl`, `reward_contact`, `reward_survive` 等官方 reward 项（禁止）  
  - `x_position`, `y_position`, `distance_from_origin` 等位置信息（禁止）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs` (当前步 27 维观测)  
- `action` (当前步输出 8 维扭矩)  
- `next_obs` (下一步 27 维观测)  
- `info` 中经本卡片允许的字段（当前为无）  
- `training_progress` 仅在 prompt 明确允许时用作调度因子，不得作为主要信号

禁止使用：
- `original_reward`（掩码且明确禁止）  
- 任何未声明的 `info` 字段  
- 任何未在本卡片声明的 obs 切片（如假设存在隐藏的 x/y 坐标）

## 7. 可用于奖励函数的信号
- position:  
  - `body_z` (obs[0])，身体高度  
  - 四元数姿态 (obs[1:5])，可计算 `body_up_z = 1 - 2*(quat_x² + quat_y²)` 得到躯干对世界 z 轴的投影（1.0 为完全直立）  
  - 8 个关节角度 (obs[5:13])
- velocity:  
  - 前向速度 `body_x_velocity` (obs[13])  
  - 横向速度 `body_y_velocity` (obs[14])  
  - 竖直速度 `body_z_velocity` (obs[15])  
  - 角速度 `body_roll/pitch/yaw_velocity` (obs[16-18])  
  - 8 个关节角速度 (obs[19:27])
- orientation: 可由 obs[1:5] 推导 body_up_z 等
- contact: **无接触力信息**，无法使用
- action/engine:  
  - 动作向量 `action` (8D) 本身，可用于扭矩惩罚或平滑惩罚  
  - 若需动作变化，需在外部保存上一动作状态，但不鼓励在无状态奖励函数中引入全局记忆

## 8. 不确定或不可用的信号
- 全局 x 坐标、y 坐标及累积前进距离（未提供，不可用）
- 任何官方 reward 分量（被掩码且禁止）
- 接触力、触地标志、脚部滑移等（未提供）
- 成功/失败标志位（info 为空）
- 直立角度 θ 的 sin/cos 形式（需要自行转换四元数）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: multi_legged_body_locomotion
control_type: continuous
morphology:
  body_type: quadruped (四足，每条腿有髋-踝两个关节)
  actuator_type: torque-controlled (8 个连续力矩关节)
  contact_structure: 足式，有地面接触但无接触力信息
primary_objectives:
  - 最大化前进速度 (body_x_velocity)
  - 保持身体高度在 (0.2, 1.0) 范围内，避免摔倒或弹跳过高
  - 维持躯干直立（body_up_z 尽量接近 1.0）
secondary_objectives:
  - 抑制横向运动和偏航，保持直线前进
  - 降低关节力矩（能耗）
  - 平稳动作，避免高频抖动
main_failure_risks:
  - 向前倾斜摔倒（高度过低）
  - 过度依赖后腿导致前腿抬起过高甚至翻滚
  - 侧向漂移大，偏离前进方向
  - 高扭矩快速切换导致关节震荡或数值溢出
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: `forward_velocity`
  - purpose: 鼓励持续向前运动，这是任务的核心目标
  - why_required: 不加入前进激励，策略只会学习站立或极慢运动
  - usable_signals: [`body_x_velocity` (obs[13])]
  - risks: 若权重过高可能导致忽视高度约束而摔倒；需与 height 角色配合

- role_id: `height_survival`
  - purpose: 惩罚身体高度接近或超出边界，维持安全范围
  - why_required: 高度是终止条件，必须让策略学会在高度内运动
  - usable_signals: [`body_z` (obs[0])]
  - risks: 仅使用一个硬边界值可能导致稀疏梯度；推荐用中心区域奖励 + 边界惩罚

- role_id: `upright_orientation`
  - purpose: 使躯干尽量朝上，防止翻滚
  - why_required: 如果不约束姿态，四足可能倒立或侧翻完成前进（不稳定）
  - usable_signals: [四元数 obs[1:5]，可计算 `body_up_z`]
  - risks: 四足前进时允许轻微前倾，过度惩罚可能抑制步态；应允许 body_up_z 在 0.8~1.0 间自由

### 10.2 条件职责 conditional_roles
- role_id: `lateral_and_yaw_penalty`
  - purpose: 抑制横向速度和偏航角速度，保证直线前进
  - condition_to_use: 观察到策略出现明显侧向漂移或原地打转时启用；也可作为默认软约束
  - usable_signals: [`body_y_velocity` (obs[14]), `body_yaw_velocity` (obs[18])]
  - risks: 过强约束可能限制自然步态的轻微横向摆动，应使用小系数

- role_id: `action_smoothness`
  - purpose: 减少相邻动作的剧烈变化，降低关节抖动
  - condition_to_use: 若动作出现高频振荡，可从训练中期加入（需存储上一动作，或通过joint velocity间接约束）
  - usable_signals: [action (当前帧，需与上一帧比较)，或使用关节角速度 (obs[19:27]) 作为代理]
  - risks: 需要存储 history，若使用关节角速度平方则可避免状态，但惩罚的是速度而非加速度变化

- role_id: `torque_cost`
  - purpose: 降低能耗，鼓励高效步态
  - condition_to_use: 通常在训练后期或作为弱正则项加入，以免干扰主要目标
  - usable_signals: [action (8D 扭矩)]
  - risks: 过大的惩罚会导致机器人无力前进，必须极低权重

### 10.3 慎用/禁用职责 avoid_roles
- role_id: `global_progress_reward`
  - reason: 需要 x 或 y 位置增量，但观测中缺失，无法实现
  - forbidden_or_missing_signals: [x_position, y_position]

- role_id: `contact_force_penalty`
  - reason: 接触力信息未提供，无法计算足端受力
  - forbidden_or_m



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





# Fresh Restart Evidence

- target_score: 2000.000
- best_score_so_far: 13.550

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| action_penalty + forward_gated + lateral_penalty | 3 | 13.550 | -464.620 | unsolved |
| action_penalty + forward_gated | 1 | 2.910 | 2.910 | unsolved |
| forward_progress + lateral_penalty | 2 | -174.240 | -174.240 | unsolved |
| action_penalty + health_gate + lateral_penalty + progress | 1 | -381.250 | -381.250 | unsolved |

## Previous interventions

- No structured intervention fields were available in the historical responses.

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
