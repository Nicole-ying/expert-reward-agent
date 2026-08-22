# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
这是一个 3D 四足机器人连续控制任务。机器人拥有四条腿、八个力矩控制关节，需要在保持身体直立且高度处于健康范围的前提下，尽可能稳定地向前行走或奔跑。主要目标是持续、快速的**前进运动**，而非仅仅保持平衡或存活。次要目标可包括维持身体姿态稳定、动作平滑、能量高效，但这些都服务于前进这一核心目标。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 任务的核心驱动力是让机器人在连续控制下产生前进运动，没有明确的终点或抓取目标；附属的姿态、高度约束属于健康条件，不是平行主目标，因此符合 locomotion_continuous_control 类型。

dynamics_subtype: multi_legged_body_locomotion  
说明：多足身体在地面上的持续步态推进，具有高维身体、关节力矩驱动、接触不直接可测的特点。

## 3. 观察空间 observation_space
- type: Box
- shape: (27,)
- dtype: float32（推断）
- 维度含义表（索引 0~26）

| obs index | name | meaning | reward_usable |
|---|---|---|---|
| 0 | body_z | 机器人主体的垂直高度 | true |
| 1 | quat_w | 身体方向四元数实部 | true |
| 2 | quat_x | 身体方向四元数虚部 x | true |
| 3 | quat_y | 身体方向四元数虚部 y | true |
| 4 | quat_z | 身体方向四元数虚部 z | true |
| 5 | joint_1_angle | 第 1 髋关节角度 | true |
| 6 | joint_2_angle | 第 1 踝关节角度 | true |
| 7 | joint_3_angle | 第 2 髋关节角度 | true |
| 8 | joint_4_angle | 第 2 踝关节角度 | true |
| 9 | joint_5_angle | 第 3 髋关节角度 | true |
| 10 | joint_6_angle | 第 3 踝关节角度 | true |
| 11 | joint_7_angle | 第 4 髋关节角度 | true |
| 12 | joint_8_angle | 第 4 踝关节角度 | true |
| 13 | body_x_velocity | 世界 x 方向前进速度 | true |
| 14 | body_y_velocity | 世界 y 方向横向速度 | true |
| 15 | body_z_velocity | 垂直速度 | true |
| 16 | body_roll_velocity | 滚转角速度 | true |
| 17 | body_pitch_velocity | 俯仰角速度 | true |
| 18 | body_yaw_velocity | 偏航角速度 | true |
| 19 | joint_1_velocity | 第 1 髋关节角速度 | true |
| 20 | joint_2_velocity | 第 1 踝关节角速度 | true |
| 21 | joint_3_velocity | 第 2 髋关节角速度 | true |
| 22 | joint_4_velocity | 第 2 踝关节角速度 | true |
| 23 | joint_5_velocity | 第 3 髋关节角速度 | true |
| 24 | joint_6_velocity | 第 3 踝关节角速度 | true |
| 25 | joint_7_velocity | 第 4 髋关节角速度 | true |
| 26 | joint_8_velocity | 第 4 踝关节角速度 | true |

额外可用派生：  
- body_up_z = 1 - 2*(quat_x² + quat_y²)，范围 [-1,1]，1 表示完全直立，可用于姿态奖励。
- 所有关节角度、速度可用于动作平滑或关节姿态惩罚。

## 4. 动作空间 action_space
- type: Box
- shape: (8,)
- continuous: true
- bounds: [-1.0, 1.0] per joint（对应标准化力矩）

| action dim | name | meaning |
|---|---|---|
| 0 | hip_1_torque | 第 1 髋关节力矩 |
| 1 | ankle_1_torque | 第 1 踝关节力矩 |
| 2 | hip_2_torque | 第 2 髋关节力矩 |
| 3 | ankle_2_torque | 第 2 踝关节力矩 |
| 4 | hip_3_torque | 第 3 髋关节力矩 |
| 5 | ankle_3_torque | 第 3 踝关节力矩 |
| 6 | hip_4_torque | 第 4 髋关节力矩 |
| 7 | ankle_4_torque | 第 4 踝关节力矩 |

## 5. step 与终止条件分析

### 5.1 终止模式
- **success-like termination**: 无显式成功终止标志。
- **failure-like termination**:  
  - body_height_outside_healthy_range: 主体垂直高度 ≤ 0.2 或 ≥ 1.0 时立即终止（可分别视为摔倒或腾空失控）。  
  - state_value_outside_finite_range: 任意状态值变为 NaN 或无穷大时终止（数值崩溃）。
- **ambiguous termination**:  
  - truncated = time_limit_reached，仅代表时间耗尽，不能直接诠释为成功或失败。
- **truncation**: 由时间限制触发。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false  
  （虽然终止原因可推断为高度越界或数值异常，但在 compute_reward 接口中无法获取 terminated 标志，只能通过 next_obs 的有限信息间接判断。）
- allowed_info_fields: []（本環境不允许在 reward 中使用任何 info 字段）
- forbidden_or_uncertain_info_fields:  
  reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin 等均不可用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs（27维）
- action（8维）
- next_obs（27维）
- training_progress（仅在 prompt 明确允许时）
- info 中明确允许的字段（本环境 **无**）

禁止使用：
- original_reward（被强制遮蔽）
- official_reward
- 未声明的 info 字段（全部 info 字段均不可用）
- 未声明的 obs/next_obs 切片（但可使用派生量，如 body_up_z）
- 任何来自环境内部的 x/y 世界坐标、接触力等

## 7. 可用于奖励函数的信号
- **position**:  
  - body_z (高度，可做高度保持奖励)  
  - 四元数 → 直立度 body_up_z  
  - 关节角度（可用于姿态正则化）
- **velocity**:  
  - body_x_velocity (世界 x 方向前进速度，核心前进信号)  
  - body_y_velocity, body_z_velocity（横向、垂直速度，可用于惩罚非前进方向运动）
  - 身体角速度 (roll, pitch, yaw) 及关节角速度（可用于平稳性惩罚）
- **orientation**:  
  - 通过四元数计算直立即时状态
- **contact**: 无（此环境无接触力信息）
- **action/engine**:  
  - 8 个关节力矩（可用于动作幅度惩罚、平滑性惩罚）
- **other**:  
  - next_obs 与 obs 的差分可用于瞬时变化量。

## 8. 不确定或不可用的信号
- 世界坐标 x, y（不可用）
- 前进距离或基准位置（不可用）
- 接触力 / 地面反作用力（不可用）
- 显式存活奖励、控制代价等原环境奖励分量（不可用）
- terminated 标志（在 reward 函数签名中未提供，亦不可从 info 获取）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: multi_legged_body_locomotion
control_type: continuous
morphology:
  body_type: quadruped
  actuator_type: torque_controlled_joints (8 DoF)
  contact_structure: four legs, no direct contact force observable
primary_objectives:
  - maximize forward velocity (body_x_velocity) over episode
secondary_objectives:
  - maintain upright posture (body_up_z close to 1)
  - keep body height stable within healthy range (avoid approaching 0.2 or 1.0)
  - minimize lateral/vertical velocity and excessive angular velocity
  - reduce energy consumption via moderate joint torques
main_failure_risks:
  - falling over (body height drops below 0.2)
  - launching or unstable jumping (body height exceeds 1.0)
  - instability causing NaN/inf states
  - learning to stand still without forward progress
  - excessive joint oscillations or jerky motion
```

## 10. 奖励职责拆解 reward_role_decomposition

### 10.1 主职责 mandatory_roles
- **role_id**: forward_progress  
  purpose: 鼓励机器人持续沿 x 方向前进，是任务核心驱动源。  
  why_required: 任务描述明确要求“walk or run forward”，且无其他到达性目标。  
  usable_signals: [body_x_velocity (obs[13])]  
  risks:  
    - 若奖励过强可能导致机器人为求高速而姿态失控；  
    - 需配合姿态/高度约束避免策略学习到跳跃前进的劣质行为。

### 10.2 条件职责 conditional_roles
- **role_id**: upright_posture  
  condition_to_use: 当前进奖励已存在前提下，建议始终施加，以过滤姿态崩溃的策略。  
  usable_signals: [body_up_z (由 quat_x, quat_y 计算)]  
  risks: 过度强调直立可能降低步态的多样性，抑制快速奔跑时的自然身体摆动。

- **role_id**: height_stability  
  condition_to_use: 建议始终施加，但可设为弱奖励，仅当高度接近阈值（如 <0.3 或 >0.9）时才显著惩罚。  
  usable_signals: [body_z (obs[0])]  
  risks: 与前进奖励可能冲突，若惩罚过重会阻碍探索；最好设计为在安全范围外有边际增加。

- **role_id**: energy_efficiency  
  condition_to_use: 在训练早期可关闭或权重较低，后期或需要稳定表现时加入，以减少关节力矩消耗。  
  usable_signals: [action (8维扭矩)]  
  risks: 过度惩罚动作幅度会压制有效步态的生成，导致前进缓慢。

- **role_id**: lateral_vertical_penalty  
  condition_to_use: 可在已有前进奖励时作为辅助，惩罚非前进方向的速度。  
  usable_signals: [body_y_velocity, body_z_velocity (obs[14], obs[15])]  
  risks: 极其微小的横向速度是四足步行固有的，过度惩罚会造成步态僵硬。

- **role_id**: angular_motion_penalty  
  condition_to_use: 若观察到翻滚或剧烈旋转时可单独启用，一般作为稳定化辅助项。  
  usable_signals: [body_roll_velocity, body_pitch_velocity, body_yaw_velocity (obs[16~18]), 各关节速度]  
  risks: 微小转动是不可避免的，需设置死区或平方窗。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id**: survival_bonus  
  reason: 环境没有存活奖励分量且无法获取终止标志，按时间步给予固定存活奖励会直接破坏前进任务的方向性，鼓励原地不动。  
  forbidden_or_missing_signals: [terminated, info['reward_survive']]

- **role_id**: goal_reaching  
  reason: 任务无固定目标点，距离或位置信息不可用，也无法定义成功，不应采用基于坐标的到达奖励。  
  forbidden_or_missing_signals: [x_position, y_position, distance_from_origin]

- **role_id**: contact_smoothness  
  reason: 接触力信息完全不可观测，无法设计基于足端接触的奖励。  
  forbidden_or_missing_signals: [contact_force]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_progress | body_x_velocity | x_position, distance | dense_state_signal, linear_clip, soft_scaling | 可以直接返回值或经过缩放，需确保正值可解释。 |
| upright_posture | quat_x, quat_y (→ body_up_z) | torso_orientation_matrix | bounded_signal, linear_to_one_reward | 建议映射到接近1的奖励（如 0.5*(body_up_z+



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

